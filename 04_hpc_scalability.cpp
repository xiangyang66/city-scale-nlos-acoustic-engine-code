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
#include <cstring>

const int Q = 19;
const double cs2 = 1.0 / 3.0;
const int ex[Q] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
const int ey[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
const int ez[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
const double w[Q] =
{
    1.0 / 3.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};
const int inv_q[Q] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};

enum CellType : uint8_t { FLUID = 0, SOLID = 1 };

struct UnitConverter
{
    double dx_phys = 0.125;
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

struct LBM
{
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0, size = 1;

    int dims[3] = {0, 0, 0};
    int periods[3] = {0, 0, 1};
    int coords[3] = {0, 0, 0};
    int nbr[6] = {MPI_PROC_NULL};

    double X_MIN = -13.0, X_MAX = 13.0;
    double Y_MIN = -13.0, Y_MAX = 13.0;
    double Z_MIN = -13.0, Z_MAX = 13.0;

    int NXg, NYg, NZg;
    int nx = 0, ny = 0, nz = 0;
    int nx_t = 0, ny_t = 0, nz_t = 0;
    int off_x = 0, off_y = 0, off_z = 0;

    UnitConverter units;

    double Amp = 0.01;
    double b_param = 2.0;
    double omega = M_PI / 15.0;

    double cyl_R = 4.0;

    double src_y = -6.0;

    double output_scale = 100.0;

    std::vector<double> f_curr, f_next, rho, sponge_sigma;
    std::vector<uint8_t> mask;
    std::vector<double> sL, rL, sR, rR, sF, rF, sB, rB, sD, rD, sU, rU;

    struct Probe
    {
        int id;
        double angle_deg;
        double x, y, z;
    };
    std::vector<Probe> circle_probes;

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
        if (NZg < 3) NZg = 3;

        MPI_Dims_create(size, 3, dims);
        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart);
        MPI_Cart_coords(cart, rank, 3, coords);
        MPI_Cart_shift(cart, 0, 1, &nbr[0], &nbr[1]);
        MPI_Cart_shift(cart, 1, 1, &nbr[2], &nbr[3]);
        MPI_Cart_shift(cart, 2, 1, &nbr[4], &nbr[5]);

        auto block_low = [&](int id, int p, int n)
        {
            return id * n / p;
        };
        auto block_size = [&](int id, int p, int n)
        {
            return block_low(id + 1, p, n) - block_low(id, p, n);
        };

        off_x = block_low(coords[0], dims[0], NXg);
        nx = block_size(coords[0], dims[0], NXg);
        off_y = block_low(coords[1], dims[1], NYg);
        ny = block_size(coords[1], dims[1], NYg);
        off_z = block_low(coords[2], dims[2], NZg);
        nz = block_size(coords[2], dims[2], NZg);

        nx_t = nx + 2;
        ny_t = ny + 2;
        nz_t = nz + 2;
        size_t Nloc = (size_t)nx_t * ny_t * nz_t;

        f_curr.resize(Nloc * Q);
        f_next.resize(Nloc * Q);
        rho.assign(Nloc, 1.0);
        sponge_sigma.assign(Nloc, 0.0);
        mask.assign(Nloc, FLUID);

        int cx = ny * nz * Q;
        int cy = (nx + 2) * nz * Q;
        int cz = (nx + 2) * (ny + 2) * Q;
        sL.resize(cx);
        rL.resize(cx);
        sR.resize(cx);
        rR.resize(cx);
        sF.resize(cy);
        rF.resize(cy);
        sB.resize(cy);
        rB.resize(cy);
        sD.resize(cz);
        rD.resize(cz);
        sU.resize(cz);
        rU.resize(cz);

        double feq0[Q];
        for (int q = 0; q < Q; ++q) feq0[q] = get_feq(q, 1.0, 0, 0, 0);
        #pragma omp parallel for
        for (size_t i = 0; i < Nloc; ++i)
        {
            for (int q = 0; q < Q; ++q) f_curr[i * Q + q] = feq0[q];
        }

        double probe_R = 60.0;
        for (int deg = 0; deg < 360; deg += 2)
        {
            double rad = deg * M_PI / 180.0;
            Probe p;
            p.id = deg;
            p.angle_deg = (double)deg;
            p.x = probe_R * std::cos(rad);
            p.y = probe_R * std::sin(rad);
            p.z = 0.0;
            circle_probes.push_back(p);
        }
    }

    void init_physics()
    {
        double sponge_start_R = 10.0;
        double sponge_thickness = 3.0;
        double damping_strength = 0.5;

        #pragma omp parallel for collapse(3)
        for (int k = 0; k < nz_t; ++k) for (int j = 0; j < ny_t; ++j) for (int i = 0; i < nx_t; ++i)
                {
                    size_t ii = loc_index(i, j, k, nx_t, ny_t);
                    double x = X_MIN + (double)(off_x + i - 1) * units.dx_phys + 0.5 * units.dx_phys;
                    double y = Y_MIN + (double)(off_y + j - 1) * units.dx_phys + 0.5 * units.dx_phys;
                    double r = std::sqrt(x * x + y * y);

                    if (r < cyl_R)
                    {
                        mask[ii] = SOLID;
                    }
                    else
                    {
                        mask[ii] = FLUID;
                        if (r > sponge_start_R)
                        {
                            double ratio = (r - sponge_start_R) / sponge_thickness;
                            if (ratio > 1.0) ratio = 1.0;
                            sponge_sigma[ii] = damping_strength * ratio * ratio;
                        }
                        else
                        {
                            sponge_sigma[ii] = 0.0;
                        }
                    }
                }
    }

    void apply_source(double t_phys, double dt_phys)
    {
        #pragma omp parallel for collapse(3)
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j) for (int i = 1; i <= nx; ++i)
                {
                    double x = X_MIN + (double)(off_x + i - 1) * units.dx_phys + 0.5 * units.dx_phys;
                    double y = Y_MIN + (double)(off_y + j - 1) * units.dx_phys + 0.5 * units.dx_phys;

                    double dist2 = x * x + (y - src_y) * (y - src_y);
                    if (dist2 > 36.0) continue;

                    double spatial = std::exp(-std::log(2.0) * dist2 / b_param);
                    double source_val = Amp * spatial * std::sin(omega * t_phys);

                    size_t idx = loc_index(i, j, k, nx_t, ny_t);
                    if (mask[idx] == FLUID)
                    {
                        for (int q = 0; q < Q; ++q)
                        {
                            f_curr[idx * Q + q] += w[q] * source_val * dt_phys;
                        }
                    }
                }
    }

    void exchange_halos()
    {
        int cx = ny * nz * Q;
        int cy = (nx + 2) * nz * Q;
        int cz = (nx + 2) * (ny + 2) * Q;
        double feq0[Q];
        for (int q = 0; q < Q; ++q) feq0[q] = get_feq(q, 1.0, 0, 0, 0);
        auto fill = [&](std::vector<double> &v, int n)
        {
            for (int i = 0; i < n / Q; ++i)for (int q = 0; q < Q; ++q)v[i * Q + q] = feq0[q];
        };
        fill(rL, cx);
        fill(rR, cx);
        fill(rF, cy);
        fill(rB, cy);
        fill(rD, cz);
        fill(rU, cz);

        int cnt = 0;
        for (int k = 1; k <= nz; ++k)for (int j = 1; j <= ny; ++j)for (int q = 0; q < Q; ++q) sL[cnt++] = f_curr[loc_index(1, j, k, nx_t, ny_t) * Q + q];
        cnt = 0;
        for (int k = 1; k <= nz; ++k)for (int j = 1; j <= ny; ++j)for (int q = 0; q < Q; ++q) sR[cnt++] = f_curr[loc_index(nx, j, k, nx_t, ny_t) * Q + q];
        MPI_Request req[4];
        MPI_Irecv(rL.data(), cx, MPI_DOUBLE, nbr[0], 10, cart, &req[0]);
        MPI_Irecv(rR.data(), cx, MPI_DOUBLE, nbr[1], 11, cart, &req[1]);
        MPI_Isend(sR.data(), cx, MPI_DOUBLE, nbr[1], 10, cart, &req[2]);
        MPI_Isend(sL.data(), cx, MPI_DOUBLE, nbr[0], 11, cart, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        cnt = 0;
        for (int k = 1; k <= nz; ++k)for (int j = 1; j <= ny; ++j)for (int q = 0; q < Q; ++q) f_curr[loc_index(0, j, k, nx_t, ny_t) * Q + q] = rL[cnt++];
        cnt = 0;
        for (int k = 1; k <= nz; ++k)for (int j = 1; j <= ny; ++j)for (int q = 0; q < Q; ++q) f_curr[loc_index(nx + 1, j, k, nx_t, ny_t) * Q + q] = rR[cnt++];

        cnt = 0;
        for (int k = 1; k <= nz; ++k)for (int i = 0; i <= nx + 1; ++i)for (int q = 0; q < Q; ++q) sF[cnt++] = f_curr[loc_index(i, 1, k, nx_t, ny_t) * Q + q];
        cnt = 0;
        for (int k = 1; k <= nz; ++k)for (int i = 0; i <= nx + 1; ++i)for (int q = 0; q < Q; ++q) sB[cnt++] = f_curr[loc_index(i, ny, k, nx_t, ny_t) * Q + q];
        MPI_Irecv(rF.data(), cy, MPI_DOUBLE, nbr[2], 20, cart, &req[0]);
        MPI_Irecv(rB.data(), cy, MPI_DOUBLE, nbr[3], 21, cart, &req[1]);
        MPI_Isend(sB.data(), cy, MPI_DOUBLE, nbr[3], 20, cart, &req[2]);
        MPI_Isend(sF.data(), cy, MPI_DOUBLE, nbr[2], 21, cart, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        cnt = 0;
        for (int k = 1; k <= nz; ++k)for (int i = 0; i <= nx + 1; ++i)for (int q = 0; q < Q; ++q) f_curr[loc_index(i, 0, k, nx_t, ny_t) * Q + q] = rF[cnt++];
        cnt = 0;
        for (int k = 1; k <= nz; ++k)for (int i = 0; i <= nx + 1; ++i)for (int q = 0; q < Q; ++q) f_curr[loc_index(i, ny + 1, k, nx_t, ny_t) * Q + q] = rB[cnt++];

        cnt = 0;
        for (int j = 0; j <= ny + 1; ++j)for (int i = 0; i <= nx + 1; ++i)for (int q = 0; q < Q; ++q) sD[cnt++] = f_curr[loc_index(i, j, 1, nx_t, ny_t) * Q + q];
        cnt = 0;
        for (int j = 0; j <= ny + 1; ++j)for (int i = 0; i <= nx + 1; ++i)for (int q = 0; q < Q; ++q) sU[cnt++] = f_curr[loc_index(i, j, nz, nx_t, ny_t) * Q + q];
        MPI_Irecv(rD.data(), cz, MPI_DOUBLE, nbr[4], 30, cart, &req[0]);
        MPI_Irecv(rU.data(), cz, MPI_DOUBLE, nbr[5], 31, cart, &req[1]);
        MPI_Isend(sU.data(), cz, MPI_DOUBLE, nbr[5], 30, cart, &req[2]);
        MPI_Isend(sD.data(), cz, MPI_DOUBLE, nbr[4], 31, cart, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);
        cnt = 0;
        for (int j = 0; j <= ny + 1; ++j)for (int i = 0; i <= nx + 1; ++i)for (int q = 0; q < Q; ++q) f_curr[loc_index(i, j, 0, nx_t, ny_t) * Q + q] = rD[cnt++];
        cnt = 0;
        for (int j = 0; j <= ny + 1; ++j)for (int i = 0; i <= nx + 1; ++i)for (int q = 0; q < Q; ++q) f_curr[loc_index(i, j, nz + 1, nx_t, ny_t) * Q + q] = rU[cnt++];
    }

    void step_pull()
    {
        double tau = 0.51;
        #pragma omp parallel for collapse(3)
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j) for (int i = 1; i <= nx; ++i)
                {
                    size_t dst = loc_index(i, j, k, nx_t, ny_t);
                    if (mask[dst] == SOLID) continue;

                    double sigma = sponge_sigma[dst];
                    bool in_sponge = (sigma > 1e-6);

                    for (int q = 0; q < Q; ++q)
                    {
                        int ip = i - ex[q], jp = j - ey[q], kp = k - ez[q];
                        size_t src = loc_index(ip, jp, kp, nx_t, ny_t);

                        if (mask[src] == SOLID)
                        {
                            f_next[dst * Q + q] = f_curr[dst * Q + inv_q[q]];
                        }
                        else
                        {
                            double rho_s = 0, ux_s = 0, uy_s = 0, uz_s = 0;
                            size_t src_idx = src * Q;
                            for (int p = 0; p < Q; ++p)
                            {
                                double val = f_curr[src_idx + p];
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

                            double feq = get_feq(q, rho_s, ux_s, uy_s, uz_s);
                            double f_post = f_curr[src_idx + q] - (f_curr[src_idx + q] - feq) / tau;

                            if (in_sponge)
                            {
                                double feq0 = get_feq(q, 1.0, 0, 0, 0);
                                f_next[dst * Q + q] = (1.0 - sigma) * f_post + sigma * feq0;
                            }
                            else
                            {
                                f_next[dst * Q + q] = f_post;
                            }
                        }
                    }
                    double r_new = 0;
                    for (int q = 0; q < Q; ++q) r_new += f_next[dst * Q + q];
                    rho[dst] = r_new;
                }
        f_curr.swap(f_next);
    }

    void run_benchmark()
    {
        int fixed_steps = 2000;

        if (rank == 0)
        {
            std::cout << "=== START BENCHMARK ===" << std::endl;
            std::cout << "Grid: " << NXg << " x " << NYg << " x " << NZg
                      << " (dx=" << units.dx_phys << ")" << std::endl;
            std::cout << "Steps=" << fixed_steps << ", MPI Ranks=" << size << std::endl;
        }

        for (int s = 0; s < 100; ++s)
        {
            double t = s * units.get_dt_phys();
            apply_source(t, units.get_dt_phys());
            exchange_halos();
            step_pull();
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int s = 0; s < fixed_steps; ++s)
        {
            double t = s * units.get_dt_phys();
            apply_source(t, units.get_dt_phys());
            exchange_halos();
            step_pull();
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(t1 - t0).count();

        if (rank == 0)
        {
            std::cout << "BENCHMARK_TIME: " << total_time << " s" << std::endl;

            double total_nodes = (double)NXg * (double)NYg * (double)NZg;
            double mlups = (total_nodes * fixed_steps) / total_time / 1e6;
            std::cout << "MLUPS: " << mlups << std::endl;
            std::cout << "=== END BENCHMARK ===" << std::endl;
        }
    }
};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    omp_set_num_threads(1);

    LBM solver;
    solver.setup_mpi(argc, argv);
    solver.init_physics();

    solver.run_benchmark();

    MPI_Finalize();
    return 0;
}