#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <filesystem>

const int Q = 19;
const int ex[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
const int ey[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
const int ez[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};

const double w[Q] =
{
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

constexpr double cs = 0.5773502691896257;
constexpr double cs2 = 1.0 / 3.0;

struct UnitConverter
{
    double dx_phys = 0.25;
    double c_phys = 1.0;
    double get_dt_phys() const
    {
        return cs * dx_phys / c_phys;
    }
};

inline size_t loc_index(int i, int j, int k, int nx_t, int ny_t)
{
    return (size_t)i + (size_t)j * nx_t + (size_t)k * nx_t * ny_t;
}

struct Solver
{
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0, size = 1;
    int dims[3] = {0, 0, 0}, periods[3] = {0, 0, 0}, coords[3] = {0, 0, 0}, nbr[6] = {MPI_PROC_NULL};

    int NXg = 400, NYg = 400, NZg = 600;
    int nx, ny, nz, nx_t, ny_t, nz_t, off_x, off_y, off_z;

    UnitConverter units;

    double Ma = 0.2;
    double uz_base_lb;
    double omega_lb;
    double amp = 0.01;

    double tau_bulk = 0.51;
    double tau_sponge_max = 0.90;
    int sponge_width = 20;

    std::vector<double> f_curr, f_next, rho, tau_field, source_spatial;

    inline double get_feq(int k, double r, double ux, double uy, double uz) const
    {
        double cu = ex[k] * ux + ey[k] * uy + ez[k] * uz;
        double u2 = ux * ux + uy * uy + uz * uz;
        return w[k] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }

    inline double get_source_forcing(int k, double source_val, double u_bg) const
    {
        double cu = ez[k] * u_bg;
        return w[k] * source_val * (1.0 + 3.0 * cu);
    }

    void setup_mpi(int argc, char **argv)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Dims_create(size, 3, dims);
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart);
        MPI_Cart_coords(cart, rank, 3, coords);

        MPI_Cart_shift(cart, 0, 1, &nbr[0], &nbr[1]);
        MPI_Cart_shift(cart, 1, 1, &nbr[2], &nbr[3]);
        MPI_Cart_shift(cart, 2, 1, &nbr[4], &nbr[5]);

        auto bl = [](int id, int p, int n)
        {
            return id * n / p;
        };

        off_x = bl(coords[0], dims[0], NXg);
        nx = bl(coords[0] + 1, dims[0], NXg) - off_x;

        off_y = bl(coords[1], dims[1], NYg);
        ny = bl(coords[1] + 1, dims[1], NYg) - off_y;

        off_z = bl(coords[2], dims[2], NZg);
        nz = bl(coords[2] + 1, dims[2], NZg) - off_z;

        nx_t = nx + 2;
        ny_t = ny + 2;
        nz_t = nz + 2;

        size_t Nloc = (size_t)nx_t * ny_t * nz_t;

        f_curr.assign(Nloc * Q, 0.0);
        f_next.assign(Nloc * Q, 0.0);
        rho.assign(Nloc, 1.0);
        tau_field.assign(Nloc, tau_bulk);
        source_spatial.assign(Nloc, 0.0);

        uz_base_lb = Ma * cs;
        omega_lb = 2.0 * M_PI / 30.0;

        if (rank == 0)
        {
            std::cout << "--- Doppler (Uniform Flow) LBM ---\n"
                      << "Grid: " << NXg << "x" << NYg << "x" << NZg
                      << "  Ma=" << Ma
                      << "  tau_bulk=" << tau_bulk
                      << "  tau_sponge_max=" << tau_sponge_max
                      << "  sponge_width=" << sponge_width
                      << std::endl;
        }
    }

    void init_physics()
    {
        double cx = NXg / 2.0, cy = NYg / 2.0, cz = NZg / 2.0;
        double dx = units.dx_phys;

        double sigma = dx;

        #pragma omp parallel for collapse(3)
        for (int k = 0; k < nz_t; ++k)
        {
            for (int j = 0; j < ny_t; ++j)
            {
                for (int i = 0; i < nx_t; ++i)
                {
                    int gx = off_x + i - 1, gy = off_y + j - 1, gz = off_z + k - 1;
                    size_t idx = loc_index(i, j, k, nx_t, ny_t);

                    int d = std::min({gx, NXg - 1 - gx, gy, NYg - 1 - gy, gz, NZg - 1 - gz});
                    if (d >= 0 && d < sponge_width)
                    {
                        double s = (double)(sponge_width - d) / (double)sponge_width;
                        double s2 = s * s;
                        tau_field[idx] = tau_bulk + (tau_sponge_max - tau_bulk) * s2;
                    }
                    else
                    {
                        tau_field[idx] = tau_bulk;
                    }

                    for (int q = 0; q < Q; ++q)
                        f_curr[idx * Q + q] = get_feq(q, 1.0, 0.0, 0.0, uz_base_lb);

                    double rx = (gx - cx) * dx;
                    double ry = (gy - cy) * dx;
                    double rz = (gz - cz) * dx;
                    double r2 = rx * rx + ry * ry + rz * rz;
                    source_spatial[idx] = amp * std::exp(-r2 / (2.0 * sigma * sigma));
                }
            }
        }
    }

    void fill_halo_copy()
    {
        if (nbr[0] == MPI_PROC_NULL)
        {
            for (int k = 0; k < nz_t; ++k)
                for (int j = 0; j < ny_t; ++j)
                {
                    size_t h = loc_index(0, j, k, nx_t, ny_t) * Q;
                    size_t in = loc_index(1, j, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[h + q] = f_curr[in + q];
                }
        }
        if (nbr[1] == MPI_PROC_NULL)
        {
            for (int k = 0; k < nz_t; ++k)
                for (int j = 0; j < ny_t; ++j)
                {
                    size_t h = loc_index(nx + 1, j, k, nx_t, ny_t) * Q;
                    size_t in = loc_index(nx, j, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[h + q] = f_curr[in + q];
                }
        }
        if (nbr[2] == MPI_PROC_NULL)
        {
            for (int k = 0; k < nz_t; ++k)
                for (int i = 0; i < nx_t; ++i)
                {
                    size_t h = loc_index(i, 0, k, nx_t, ny_t) * Q;
                    size_t in = loc_index(i, 1, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[h + q] = f_curr[in + q];
                }
        }
        if (nbr[3] == MPI_PROC_NULL)
        {
            for (int k = 0; k < nz_t; ++k)
                for (int i = 0; i < nx_t; ++i)
                {
                    size_t h = loc_index(i, ny + 1, k, nx_t, ny_t) * Q;
                    size_t in = loc_index(i, ny, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[h + q] = f_curr[in + q];
                }
        }
        if (nbr[4] == MPI_PROC_NULL)
        {
            for (int j = 0; j < ny_t; ++j)
                for (int i = 0; i < nx_t; ++i)
                {
                    size_t h = loc_index(i, j, 0, nx_t, ny_t) * Q;
                    size_t in = loc_index(i, j, 1, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[h + q] = f_curr[in + q];
                }
        }
        if (nbr[5] == MPI_PROC_NULL)
        {
            for (int j = 0; j < ny_t; ++j)
                for (int i = 0; i < nx_t; ++i)
                {
                    size_t h = loc_index(i, j, nz + 1, nx_t, ny_t) * Q;
                    size_t in = loc_index(i, j, nz, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[h + q] = f_curr[in + q];
                }
        }
    }

    void exchange_halos()
    {
        int cx = ny * nz * Q, cy = nx_t * nz * Q, cz = nx_t * ny_t * Q;
        std::vector<double> sL(cx), rL(cx), sR(cx), rR(cx),
            sF(cy), rF(cy), sB(cy), rB(cy),
            sD(cz), rD(cz), sU(cz), rU(cz);
        MPI_Request reqs[4];
        int p;

        p = 0;
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
            {
                size_t bL = loc_index(1, j, k, nx_t, ny_t) * Q;
                size_t bR = loc_index(nx, j, k, nx_t, ny_t) * Q;
                for (int q = 0; q < Q; ++q)
                {
                    sL[p] = f_curr[bL + q];
                    sR[p] = f_curr[bR + q];
                    p++;
                }
            }

        MPI_Irecv(rL.data(), cx, MPI_DOUBLE, nbr[0], 10, cart, &reqs[0]);
        MPI_Irecv(rR.data(), cx, MPI_DOUBLE, nbr[1], 11, cart, &reqs[1]);
        MPI_Isend(sR.data(), cx, MPI_DOUBLE, nbr[1], 10, cart, &reqs[2]);
        MPI_Isend(sL.data(), cx, MPI_DOUBLE, nbr[0], 11, cart, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        if (nbr[0] != MPI_PROC_NULL)
        {
            p = 0;
            for (int k = 1; k <= nz; ++k)
                for (int j = 1; j <= ny; ++j)
                {
                    size_t b = loc_index(0, j, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = rL[p++];
                }
        }
        if (nbr[1] != MPI_PROC_NULL)
        {
            p = 0;
            for (int k = 1; k <= nz; ++k)
                for (int j = 1; j <= ny; ++j)
                {
                    size_t b = loc_index(nx + 1, j, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = rR[p++];
                }
        }

        p = 0;
        for (int k = 1; k <= nz; ++k)
            for (int i = 0; i < nx_t; ++i)
            {
                size_t bF = loc_index(i, 1, k, nx_t, ny_t) * Q;
                size_t bB = loc_index(i, ny, k, nx_t, ny_t) * Q;
                for (int q = 0; q < Q; ++q)
                {
                    sF[p] = f_curr[bF + q];
                    sB[p] = f_curr[bB + q];
                    p++;
                }
            }

        MPI_Irecv(rF.data(), cy, MPI_DOUBLE, nbr[2], 20, cart, &reqs[0]);
        MPI_Irecv(rB.data(), cy, MPI_DOUBLE, nbr[3], 21, cart, &reqs[1]);
        MPI_Isend(sB.data(), cy, MPI_DOUBLE, nbr[3], 20, cart, &reqs[2]);
        MPI_Isend(sF.data(), cy, MPI_DOUBLE, nbr[2], 21, cart, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        if (nbr[2] != MPI_PROC_NULL)
        {
            p = 0;
            for (int k = 1; k <= nz; ++k)
                for (int i = 0; i < nx_t; ++i)
                {
                    size_t b = loc_index(i, 0, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = rF[p++];
                }
        }
        if (nbr[3] != MPI_PROC_NULL)
        {
            p = 0;
            for (int k = 1; k <= nz; ++k)
                for (int i = 0; i < nx_t; ++i)
                {
                    size_t b = loc_index(i, ny + 1, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = rB[p++];
                }
        }

        p = 0;
        for (int j = 0; j < ny_t; ++j)
            for (int i = 0; i < nx_t; ++i)
            {
                size_t bD = loc_index(i, j, 1, nx_t, ny_t) * Q;
                size_t bU = loc_index(i, j, nz, nx_t, ny_t) * Q;
                for (int q = 0; q < Q; ++q)
                {
                    sD[p] = f_curr[bD + q];
                    sU[p] = f_curr[bU + q];
                    p++;
                }
            }

        MPI_Irecv(rD.data(), cz, MPI_DOUBLE, nbr[4], 30, cart, &reqs[0]);
        MPI_Irecv(rU.data(), cz, MPI_DOUBLE, nbr[5], 31, cart, &reqs[1]);
        MPI_Isend(sU.data(), cz, MPI_DOUBLE, nbr[5], 30, cart, &reqs[2]);
        MPI_Isend(sD.data(), cz, MPI_DOUBLE, nbr[4], 31, cart, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        if (nbr[4] != MPI_PROC_NULL)
        {
            p = 0;
            for (int j = 0; j < ny_t; ++j)
                for (int i = 0; i < nx_t; ++i)
                {
                    size_t b = loc_index(i, j, 0, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = rD[p++];
                }
        }
        if (nbr[5] != MPI_PROC_NULL)
        {
            p = 0;
            for (int j = 0; j < ny_t; ++j)
                for (int i = 0; i < nx_t; ++i)
                {
                    size_t b = loc_index(i, j, nz + 1, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = rU[p++];
                }
        }

        fill_halo_copy();
    }

    inline void macroscopic(size_t idx, double &r, double &ux, double &uy, double &uz) const
    {
        r = 0.0;
        ux = uy = uz = 0.0;
        size_t base = idx * Q;
        for (int p = 0; p < Q; ++p)
        {
            double v = f_curr[base + p];
            r += v;
            ux += v * ex[p];
            uy += v * ey[p];
            uz += v * ez[p];
        }
        if (r > 1e-12)
        {
            ux /= r;
            uy /= r;
            uz /= r;
        }
        else
        {
            r = 1.0;
            ux = 0.0;
            uy = 0.0;
            uz = uz_base_lb;
        }
    }

    void step(int s)
    {
        double dt = units.get_dt_phys();
        double sin_val = std::sin(omega_lb * s) * dt;

        #pragma omp parallel for collapse(3)
        for (int k = 1; k <= nz; ++k)
        {
            for (int j = 1; j <= ny; ++j)
            {
                for (int i = 1; i <= nx; ++i)
                {
                    size_t dst = loc_index(i, j, k, nx_t, ny_t);
                    double r_n = 0.0;

                    for (int q = 0; q < Q; ++q)
                    {
                        size_t src = loc_index(i - ex[q], j - ey[q], k - ez[q], nx_t, ny_t);

                        double tau_s = tau_field[src];
                        double discrete_correction = (1.0 - 0.5 / tau_s);

                        double s_amp = source_spatial[src] * sin_val;

                        double r_s, ux_s, uy_s, uz_s;
                        macroscopic(src, r_s, ux_s, uy_s, uz_s);

                        double feq = get_feq(q, r_s, ux_s, uy_s, uz_s);

                        double force = get_source_forcing(q, s_amp, uz_base_lb);

                        double f_old = f_curr[src * Q + q];
                        double f_new = f_old - (f_old - feq) / tau_s
                                       + force * discrete_correction;

                        f_next[dst * Q + q] = f_new;
                        r_n += f_new;
                    }

                    rho[dst] = r_n;
                }
            }
        }

        f_curr.swap(f_next);
    }

    void write_dat(int s, double time)
    {
        int tx = NXg / 2, ty = NYg / 2;
        std::vector<double> lz(NZg, 0.0), gz(NZg, 0.0);

        if (tx >= off_x && tx < off_x + nx && ty >= off_y && ty < off_y + ny)
        {
            for (int k = 1; k <= nz; ++k)
            {
                size_t idx = loc_index(tx - off_x + 1, ty - off_y + 1, k, nx_t, ny_t);
                lz[off_z + k - 1] = rho[idx] - 1.0;
            }
        }

        MPI_Reduce(lz.data(), gz.data(), NZg, MPI_DOUBLE, MPI_SUM, 0, cart);

        if (rank == 0)
        {
            std::ofstream ofs("output_mpi/centerline_t" + std::to_string(time) + ".dat");
            ofs << std::fixed << std::setprecision(10);
            for (int k = 0; k < NZg; ++k)
                ofs << (k - NZg / 2.0) * units.dx_phys << " " << gz[k] << "\n";
        }
    }

    void write_vtk(int s, double time)
    {
        std::vector<double> lbuf((size_t)nx * ny * nz);
        int c = 0;
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i)
                    lbuf[c++] = rho[loc_index(i, j, k, nx_t, ny_t)] - 1.0;

        if (rank != 0)
        {
            MPI_Send(lbuf.data(), (int)lbuf.size(), MPI_DOUBLE, 0, 99, cart);
        }
        else
        {
            std::vector<double> gbuf((size_t)NXg * NYg * NZg, 0.0);
            c = 0;

            for (int k = 0; k < nz; ++k)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i)
                        gbuf[(size_t)(off_z + k) * NXg * NYg + (size_t)(off_y + j) * NXg + (off_x + i)] = lbuf[c++];

            for (int p = 1; p < size; ++p)
            {
                int co[3];
                MPI_Cart_coords(cart, p, 3, co);

                auto bl = [](int id, int pn, int n)
                {
                    return id * n / pn;
                };

                int ox = bl(co[0], dims[0], NXg), sx = bl(co[0] + 1, dims[0], NXg) - ox;
                int oy = bl(co[1], dims[1], NYg), sy = bl(co[1] + 1, dims[1], NYg) - oy;
                int oz = bl(co[2], dims[2], NZg), sz = bl(co[2] + 1, dims[2], NZg) - oz;

                std::vector<double> tmp((size_t)sx * sy * sz);
                MPI_Recv(tmp.data(), (int)tmp.size(), MPI_DOUBLE, p, 99, cart, MPI_STATUS_IGNORE);

                int tc = 0;
                for (int k = 0; k < sz; ++k)
                    for (int j = 0; j < sy; ++j)
                        for (int i = 0; i < sx; ++i)
                            gbuf[(size_t)(oz + k) * NXg * NYg + (size_t)(oy + j) * NXg + (ox + i)] = tmp[tc++];
            }

            std::ofstream ofs("output_mpi/field_t" + std::to_string(time) + ".vtk");
            ofs << "# vtk DataFile Version 3.0\nLBM_Doppler\nASCII\nDATASET STRUCTURED_POINTS\n";
            ofs << "DIMENSIONS " << NXg << " " << NYg << " " << NZg << "\n";
            ofs << "ORIGIN " << -NXg * units.dx_phys / 2.0 << " "
                << -NYg * units.dx_phys / 2.0 << " "
                << -NZg * units.dx_phys / 2.0 << "\n";
            ofs << "SPACING " << units.dx_phys << " " << units.dx_phys << " " << units.dx_phys << "\n";
            ofs << "POINT_DATA " << (size_t)NXg * NYg * NZg << "\nSCALARS Pressure float 1\nLOOKUP_TABLE default\n";
            for (double v : gbuf) ofs << v << "\n";
        }
    }

    void run()
    {
        double dt = units.get_dt_phys();
        int max_s = (int)(110.0 / dt);
        int target_s = (int)(100.0 / dt);
        int vtk_gap = std::max(1, max_s / 5);

        for (int s = 0; s <= max_s; ++s)
        {
            double cur_t = s * dt;

            if (s % vtk_gap == 0 || s == target_s)
            {
                write_vtk(s, cur_t);
                write_dat(s, cur_t);
                if (rank == 0) std::cout << "Step " << s << " (T=" << cur_t << ") Output written.\n";
            }

            exchange_halos();
            step(s);
        }
    }
};

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    if (r == 0) std::filesystem::create_directories("output_mpi");

    Solver s;
    s.setup_mpi(argc, argv);
    s.init_physics();
    s.run();

    MPI_Finalize();
    return 0;
}