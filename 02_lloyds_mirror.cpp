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
const double cs2 = 1.0 / 3.0;
const int ex[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
const int ey[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
const int ez[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
const int inv_q[Q] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
const double w[Q] =
{
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

enum CellType : uint8_t { FLUID = 0, SOLID = 1 };

struct UnitConverter
{
    double dx_phys = 0.2;
    double c_phys = 1.0;
    double get_dt_phys() const
    {
        return 0.577350269 * dx_phys / c_phys;
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
    int dims[3] = {0, 0, 0};
    int periods[3] = {0, 0, 0};
    int coords[3] = {0, 0, 0};
    int nbr[6] = {MPI_PROC_NULL};

    double X_MIN = -50.0, X_MAX = 50.0;
    double Y_MIN = -50.0, Y_MAX = 50.0;
    double Z_MIN = -10.0, Z_MAX = 50.0;

    int NXg, NYg, NZg;
    int nx = 0, ny = 0, nz = 0;
    int nx_t = 0, ny_t = 0, nz_t = 0;
    int off_x = 0, off_y = 0, off_z = 0;

    UnitConverter units;

    double omega = 0.0;
    double amp = 0.5;
    double b_param = 2.0;

    std::vector<double> f_curr, f_next, rho, tau_field;
    std::vector<double> source_spatial;
    std::vector<uint8_t> mask;

    double t_run = 0.0;

    inline double get_feq(int k, double rho_loc, double ux, double uy, double uz)
    {
        double cu = ex[k] * ux + ey[k] * uy + ez[k] * uz;
        double u2 = ux * ux + uy * uy + uz * uz;
        return w[k] * rho_loc * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }

    void setup_mpi(int, char**)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        NXg = (int)((X_MAX - X_MIN) / units.dx_phys + 0.5);
        NYg = (int)((Y_MAX - Y_MIN) / units.dx_phys + 0.5);
        NZg = (int)((Z_MAX - Z_MIN) / units.dx_phys + 0.5);

        MPI_Dims_create(size, 3, dims);
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart);
        MPI_Cart_coords(cart, rank, 3, coords);
        MPI_Cart_shift(cart, 0, 1, &nbr[0], &nbr[1]);
        MPI_Cart_shift(cart, 1, 1, &nbr[2], &nbr[3]);
        MPI_Cart_shift(cart, 2, 1, &nbr[4], &nbr[5]);

        auto block_low = [](int id, int p, int n)
        {
            return id * n / p;
        };
        auto block_len = [&](int id, int p, int n)
        {
            return block_low(id + 1, p, n) - block_low(id, p, n);
        };

        off_x = block_low(coords[0], dims[0], NXg);
        nx = block_len(coords[0], dims[0], NXg);
        off_y = block_low(coords[1], dims[1], NYg);
        ny = block_len(coords[1], dims[1], NYg);
        off_z = block_low(coords[2], dims[2], NZg);
        nz = block_len(coords[2], dims[2], NZg);

        nx_t = nx + 2;
        ny_t = ny + 2;
        nz_t = nz + 2;
        size_t Nloc = (size_t)nx_t * ny_t * nz_t;

        f_curr.resize(Nloc * Q);
        f_next.resize(Nloc * Q);
        rho.resize(Nloc);
        tau_field.resize(Nloc);
        source_spatial.resize(Nloc, 0.0);
        mask.assign(Nloc, FLUID);

        omega = 3.14159265358979323846 / 15.0;

        if (rank == 0)
        {
            std::cout << "--- Case 3.5: Source with Wall ---\n";
            std::cout << "Domain: " << NXg << "x" << NYg << "x" << NZg << "\n";
            std::cout << "Wall at Z = " << Z_MIN << "\n";
        }
    }

    void init_physics()
    {
        double tau_bulk = 0.51;
        int sponge_w = 15;

        #pragma omp parallel for collapse(3)
        for (int k = 0; k < nz_t; ++k) for (int j = 0; j < ny_t; ++j) for (int i = 0; i < nx_t; ++i)
                {
                    int gx = off_x + i - 1;
                    int gy = off_y + j - 1;
                    int gz = off_z + k - 1;
                    size_t idx = loc_index(i, j, k, nx_t, ny_t);

                    double px = X_MIN + (gx + 0.5) * units.dx_phys;
                    double py = Y_MIN + (gy + 0.5) * units.dx_phys;
                    double pz = Z_MIN + (gz + 0.5) * units.dx_phys;

                    if (gz == 0)
                    {
                        mask[idx] = SOLID;
                        rho[idx] = 1.0;
                        for (int q = 0; q < Q; ++q) f_curr[idx * Q + q] = get_feq(q, 1.0, 0, 0, 0);
                        continue;
                    }
                    else
                    {
                        mask[idx] = FLUID;
                    }

                    int dist_x = std::min(gx, NXg - 1 - gx);
                    int dist_y = std::min(gy, NYg - 1 - gy);
                    int dist_z_top = NZg - 1 - gz;

                    int dist = std::min({dist_x, dist_y, dist_z_top});

                    if (dist < sponge_w && dist >= 0)
                    {
                        double r = (double)(sponge_w - dist) / sponge_w;
                        tau_field[idx] = tau_bulk + (1.0 - tau_bulk) * r * r;
                    }
                    else
                    {
                        tau_field[idx] = tau_bulk;
                    }

                    double r2 = px * px + py * py + pz * pz;
                    source_spatial[idx] = amp * std::exp(-std::log(2.0) * r2 / b_param);

                    rho[idx] = 1.0;
                    for (int q = 0; q < Q; ++q)
                    {
                        f_curr[idx * Q + q] = get_feq(q, 1.0, 0, 0, 0);
                    }
                }
    }

    void exchange_halos()
    {
        int cx = ny * nz * Q, cy = (nx + 2) * nz * Q, cz = (nx + 2) * (ny + 2) * Q;
        std::vector<double> sL(cx), rL(cx), sR(cx), rR(cx);
        std::vector<double> sF(cy), rF(cy), sB(cy), rB(cy);
        std::vector<double> sD(cz), rD(cz), sU(cz), rU(cz);

        MPI_Request reqs[4];

        int p = 0;
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j)
            {
                size_t b = loc_index(1, j, k, nx_t, ny_t) * Q;
                for (int q = 0; q < Q; ++q) sL[p++] = f_curr[b + q];
            }
        p = 0;
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j)
            {
                size_t b = loc_index(nx, j, k, nx_t, ny_t) * Q;
                for (int q = 0; q < Q; ++q) sR[p++] = f_curr[b + q];
            }
        MPI_Irecv(rL.data(), cx, MPI_DOUBLE, nbr[0], 10, cart, &reqs[0]);
        MPI_Irecv(rR.data(), cx, MPI_DOUBLE, nbr[1], 11, cart, &reqs[1]);
        MPI_Isend(sR.data(), cx, MPI_DOUBLE, nbr[1], 10, cart, &reqs[2]);
        MPI_Isend(sL.data(), cx, MPI_DOUBLE, nbr[0], 11, cart, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        if (nbr[0] != MPI_PROC_NULL)
        {
            p = 0;
            for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j)
                {
                    size_t b = loc_index(0, j, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = rL[p++];
                }
        }
        else
        {
            for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j)
                {
                    size_t b = loc_index(0, j, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = get_feq(q, 1.0, 0, 0, 0);
                }
        }

        if (nbr[1] != MPI_PROC_NULL)
        {
            p = 0;
            for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j)
                {
                    size_t b = loc_index(nx + 1, j, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = rR[p++];
                }
        }
        else
        {
            for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j)
                {
                    size_t b = loc_index(nx + 1, j, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = get_feq(q, 1.0, 0, 0, 0);
                }
        }

        p = 0;
        for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i)
            {
                size_t b = loc_index(i, 1, k, nx_t, ny_t) * Q;
                for (int q = 0; q < Q; ++q) sF[p++] = f_curr[b + q];
            }
        p = 0;
        for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i)
            {
                size_t b = loc_index(i, ny, k, nx_t, ny_t) * Q;
                for (int q = 0; q < Q; ++q) sB[p++] = f_curr[b + q];
            }
        MPI_Irecv(rF.data(), cy, MPI_DOUBLE, nbr[2], 20, cart, &reqs[0]);
        MPI_Irecv(rB.data(), cy, MPI_DOUBLE, nbr[3], 21, cart, &reqs[1]);
        MPI_Isend(sB.data(), cy, MPI_DOUBLE, nbr[3], 20, cart, &reqs[2]);
        MPI_Isend(sF.data(), cy, MPI_DOUBLE, nbr[2], 21, cart, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        if (nbr[2] != MPI_PROC_NULL)
        {
            p = 0;
            for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i)
                {
                    size_t b = loc_index(i, 0, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = rF[p++];
                }
        }
        else
        {
            for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i)
                {
                    size_t b = loc_index(i, 0, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = get_feq(q, 1.0, 0, 0, 0);
                }
        }

        if (nbr[3] != MPI_PROC_NULL)
        {
            p = 0;
            for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i)
                {
                    size_t b = loc_index(i, ny + 1, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = rB[p++];
                }
        }
        else
        {
            for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i)
                {
                    size_t b = loc_index(i, ny + 1, k, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = get_feq(q, 1.0, 0, 0, 0);
                }
        }

        p = 0;
        for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i)
            {
                size_t b = loc_index(i, j, 1, nx_t, ny_t) * Q;
                for (int q = 0; q < Q; ++q) sD[p++] = f_curr[b + q];
            }
        p = 0;
        for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i)
            {
                size_t b = loc_index(i, j, nz, nx_t, ny_t) * Q;
                for (int q = 0; q < Q; ++q) sU[p++] = f_curr[b + q];
            }
        MPI_Irecv(rD.data(), cz, MPI_DOUBLE, nbr[4], 30, cart, &reqs[0]);
        MPI_Irecv(rU.data(), cz, MPI_DOUBLE, nbr[5], 31, cart, &reqs[1]);
        MPI_Isend(sU.data(), cz, MPI_DOUBLE, nbr[5], 30, cart, &reqs[2]);
        MPI_Isend(sD.data(), cz, MPI_DOUBLE, nbr[4], 31, cart, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        if (nbr[4] != MPI_PROC_NULL)
        {
            p = 0;
            for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i)
                {
                    size_t b = loc_index(i, j, 0, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = rD[p++];
                }
        }
        else
        {
            for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i)
                {
                    size_t b = loc_index(i, j, 0, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = get_feq(q, 1.0, 0, 0, 0);
                }
        }

        if (nbr[5] != MPI_PROC_NULL)
        {
            p = 0;
            for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i)
                {
                    size_t b = loc_index(i, j, nz + 1, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = rU[p++];
                }
        }
        else
        {
            for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i)
                {
                    size_t b = loc_index(i, j, nz + 1, nx_t, ny_t) * Q;
                    for (int q = 0; q < Q; ++q) f_curr[b + q] = get_feq(q, 1.0, 0, 0, 0);
                }
        }
    }

    void step(double current_phys_time)
    {
        double dt = units.get_dt_phys();

        #pragma omp parallel for collapse(3)
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j) for (int i = 1; i <= nx; ++i)
                {
                    size_t dst = loc_index(i, j, k, nx_t, ny_t);

                    if (mask[dst] == SOLID)
                    {
                        continue;
                    }

                    double rho_new = 0;
                    double source_val = source_spatial[dst] * std::sin(omega * current_phys_time) * dt;

                    for (int q = 0; q < Q; ++q)
                    {
                        int si = i - ex[q];
                        int sj = j - ey[q];
                        int sk = k - ez[q];
                        size_t src = loc_index(si, sj, sk, nx_t, ny_t);

                        if (mask[src] == SOLID)
                        {
                            f_next[dst * Q + q] = f_curr[dst * Q + inv_q[q]];
                            rho_new += f_next[dst * Q + q];
                        }
                        else
                        {
                            double rho_s = 0, ux_s = 0, uy_s = 0, uz_s = 0;
                            for (int p = 0; p < Q; ++p)
                            {
                                double val = f_curr[src * Q + p];
                                rho_s += val;
                                ux_s += val * ex[p];
                                uy_s += val * ey[p];
                                uz_s += val * ez[p];
                            }
                            if (rho_s > 0)
                            {
                                ux_s /= rho_s;
                                uy_s /= rho_s;
                                uz_s /= rho_s;
                            }
                            else rho_s = 1.0;

                            double feq = get_feq(q, rho_s, ux_s, uy_s, uz_s);
                            double fpost = f_curr[src * Q + q] - (f_curr[src * Q + q] - feq) / tau_field[src];

                            fpost += w[q] * source_val;

                            f_next[dst * Q + q] = fpost;
                            rho_new += fpost;
                        }
                    }
                    rho[dst] = rho_new;
                }
        f_curr.swap(f_next);
    }

    void write_y_centerline(int step, double time)
    {
        int tx = NXg / 2;
        int tz = (int)((0.0 - Z_MIN) / units.dx_phys + 0.5);

        std::vector<double> ly(NYg, 0.0);

        if (tx >= off_x && tx < off_x + nx && tz >= off_z && tz < off_z + nz)
        {
            int li = tx - off_x + 1;
            int lk = tz - off_z + 1;
            for (int j = 1; j <= ny; ++j)
            {
                size_t idx = loc_index(li, j, lk, nx_t, ny_t);
                ly[off_y + j - 1] = rho[idx];
            }
        }

        std::vector<double> gy(NYg);
        MPI_Reduce(ly.data(), gy.data(), NYg, MPI_DOUBLE, MPI_SUM, 0, cart);

        if (rank == 0)
        {
            std::ostringstream oss;
            oss << "output_mpi/centerline_y_t" << (int)(time + 0.1) << ".dat";
            std::ofstream ofs(oss.str());
            ofs << "# Y p_prime\n";
            for (int j = 0; j < NYg; ++j)
            {
                double y = Y_MIN + j * units.dx_phys + units.dx_phys / 2.0;
                double pp = (gy[j] - 1.0);
                ofs << y << " " << pp << "\n";
            }
        }
    }

    void write_vtk(int step, double time)
    {
        std::vector<double> lbuf(nx * ny * nz);
        int cnt = 0;
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i)
                    lbuf[cnt++] = rho[loc_index(i, j, k, nx_t, ny_t)];

        if (rank != 0)
        {
            MPI_Send(lbuf.data(), lbuf.size(), MPI_DOUBLE, 0, 999, cart);
        }
        else
        {
            std::vector<double> gbuf(NXg * NYg * NZg);

            cnt = 0;
            for (int k = 0; k <= nz - 1; ++k)
                for (int j = 0; j <= ny - 1; ++j)
                    for (int i = 0; i <= nx - 1; ++i)
                        gbuf[(off_z + k) * NXg * NYg + (off_y + j) * NXg + (off_x + i)] = lbuf[cnt++];

            for (int p = 1; p < size; ++p)
            {
                int c[3];
                MPI_Cart_coords(cart, p, 3, c);
                auto bl = [&](int id, int pn, int n)
                {
                    return id * n / pn;
                };
                auto bs = [&](int id, int pn, int n)
                {
                    return bl(id + 1, pn, n) - bl(id, pn, n);
                };

                int ox = bl(c[0], dims[0], NXg), sx = bs(c[0], dims[0], NXg);
                int oy = bl(c[1], dims[1], NYg), sy = bs(c[1], dims[1], NYg);
                int oz = bl(c[2], dims[2], NZg), sz = bs(c[2], dims[2], NZg);

                std::vector<double> tmp(sx * sy * sz);
                MPI_Recv(tmp.data(), tmp.size(), MPI_DOUBLE, p, 999, cart, MPI_STATUS_IGNORE);

                int tcnt = 0;
                for (int k = 0; k < sz; ++k)
                    for (int j = 0; j < sy; ++j)
                        for (int i = 0; i < sx; ++i)
                            gbuf[(oz + k) * NXg * NYg + (oy + j) * NXg + (ox + i)] = tmp[tcnt++];
            }

            std::ostringstream oss;
            oss << "output_mpi/field_case35_t" << (int)(time + 0.1) << ".vtk";
            std::ofstream ofs(oss.str());

            ofs << "# vtk DataFile Version 3.0\nLBM_Case35_Output\nASCII\nDATASET STRUCTURED_POINTS\n";
            ofs << "DIMENSIONS " << NXg << " " << NYg << " " << NZg << "\n";
            ofs << "ORIGIN " << X_MIN << " " << Y_MIN << " " << Z_MIN << "\n";
            ofs << "SPACING " << units.dx_phys << " " << units.dx_phys << " " << units.dx_phys << "\n";

            ofs << "POINT_DATA " << NXg * NYg * NZg << "\n";

            ofs << "SCALARS Rho_Perturbation float 1\nLOOKUP_TABLE default\n";
            for (double r : gbuf) ofs << (r - 1.0) << "\n";

            ofs << "SCALARS Acoustic_Pressure float 1\nLOOKUP_TABLE default\n";
            for (double r : gbuf) ofs << (r - 1.0) << "\n";
        }
        MPI_Barrier(cart);
    }

    void run_simulation()
    {
        auto t0_clk = std::chrono::high_resolution_clock::now();
        double dt = units.get_dt_phys();

        int step_t30 = (int)(30.0 / dt + 0.5);
        int step_t45 = (int)(45.0 / dt + 0.5);
        int max_step = step_t45 + 1;

        if (rank == 0) std::cout << "Targets: T=30->" << step_t30 << ", T=45->" << step_t45 << "\n";

        for (int s = 0; s <= max_step; ++s)
        {
            double time = s * dt;

            if (s == step_t30)
            {
                if (rank == 0) std::cout << "Output T=30\n";
                write_vtk(s, 30.0);
                write_y_centerline(s, 30.0);
            }
            if (s == step_t45)
            {
                if (rank == 0) std::cout << "Output T=45\n";
                write_y_centerline(s, 45.0);
            }

            exchange_halos();
            step(time);

            if (s % 50 == 0 && rank == 0) std::cout << "Step " << s << "/" << max_step << "\r" << std::flush;
        }
        t_run = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0_clk).count();
        if (rank == 0) std::cout << "\nFinished in " << t_run << " s\n";
    }
};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    if (r == 0) std::filesystem::create_directories("output_mpi");
    Solver s;
    s.setup_mpi(argc, argv);
    s.init_physics();
    s.run_simulation();
    MPI_Finalize();
    return 0;
}