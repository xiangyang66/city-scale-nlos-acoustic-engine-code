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
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};
const int inv_q[Q] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};

enum CellType : uint8_t { FLUID = 0, SOLID = 1 };

struct UnitConverter
{
    double dx_phys;
    double c_phys;
    double get_dt_phys() const
    {
        double cs_lb = std::sqrt(1.0 / 3.0);
        return cs_lb * dx_phys / c_phys;
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
    int nbr[6] = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};

    double X_MIN = -8.0, X_MAX = 8.0;
    double Y_MIN = -8.0, Y_MAX = 8.0;
    double Z_MIN = -0.2, Z_MAX = 0.2;

    int NXg, NYg, NZg;
    int nx = 0, ny = 0, nz = 0;
    int nx_t = 0, ny_t = 0, nz_t = 0;
    int off_x = 0, off_y = 0, off_z = 0;

    UnitConverter units{0.05, 1.0};

    double pulse_amp_lbm = 0.01;
    double output_scale_factor = 100.0;

    std::vector<double> f_curr;
    std::vector<double> f_next;
    std::vector<double> rho;
    std::vector<double> sponge_sigma;
    std::vector<uint8_t> mask;

    struct Probe
    {
        std::string name;
        double x, y, z;
        std::vector<std::pair<double, double>> history;
    };
    std::vector<Probe> probes;

    double t_setup = 0, t_run = 0;

    inline double get_feq(int k, double rho_loc, double ux, double uy, double uz)
    {
        double cu = ex[k] * ux + ey[k] * uy + ez[k] * uz;
        double u2 = ux * ux + uy * uy + uz * uz;
        return w[k] * rho_loc * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }

    void setup_mpi(int argc, char** argv)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        NXg = (int)((X_MAX - X_MIN) / units.dx_phys + 0.5);
        NYg = (int)((Y_MAX - Y_MIN) / units.dx_phys + 0.5);
        NZg = (int)((Z_MAX - Z_MIN) / units.dx_phys + 0.5);
        if (NZg < 3) NZg = 3;

        MPI_Dims_create(size, 3, dims);

        periods[0] = 0;
        periods[1] = 0;
        periods[2] = 1;

        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 1, &cart);
        MPI_Cart_coords(cart, rank, 3, coords);
        MPI_Cart_shift(cart, 0, 1, &nbr[0], &nbr[1]);
        MPI_Cart_shift(cart, 1, 1, &nbr[2], &nbr[3]);
        MPI_Cart_shift(cart, 2, 1, &nbr[4], &nbr[5]);

        auto block_low = [](int id, int p, int n)
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
        f_curr.assign(Nloc * Q, 0.0);
        f_next.assign(Nloc * Q, 0.0);
        rho.assign(Nloc, 1.0);
        sponge_sigma.assign(Nloc, 0.0);
        mask.assign(Nloc, FLUID);

        double feq0[Q];
        for (int q = 0; q < Q; ++q) feq0[q] = get_feq(q, 1.0, 0, 0, 0);
        #pragma omp parallel for
        for (size_t ii = 0; ii < Nloc; ++ii)
        {
            for (int q = 0; q < Q; ++q) f_curr[ii * Q + q] = feq0[q];
        }

        if (rank == 0)
        {
            std::cout << "--- LBM Cylinder V7 (Physics Calibrated) ---\n";
            std::cout << "Output Scale Factor: " << output_scale_factor << "\n";
            std::cout << "Domain: [" << X_MIN << ", " << X_MAX << "]\n";
        }

        double rad135 = 135.0 * M_PI / 180.0;
        probes =
        {
            {"A", 0.0, 1.0, 0.0},
            {"B", std::cos(rad135), std::sin(rad135), 0.0},
            {"C", -1.0, 0.0, 0.0}
        };
    }

    void init_physics()
    {
        double cyl_radius = 0.5;
        double pulse_center_x = 4.0;
        double pulse_center_y = 0.0;
        double pulse_width_b = 1.0;

        double sponge_thickness = 0.5;
        double damping_strength = 1.0;

        #pragma omp parallel for collapse(3)
        for (int k = 0; k < nz_t; ++k) for (int j = 0; j < ny_t; ++j) for (int i = 0; i < nx_t; ++i)
                {
                    size_t ii = loc_index(i, j, k, nx_t, ny_t);
                    int gx = off_x + (i - 1);
                    int gy = off_y + (j - 1);

                    double x = X_MIN + (gx + 0.5) * units.dx_phys;
                    double y = Y_MIN + (gy + 0.5) * units.dx_phys;

                    if (x * x + y * y < cyl_radius * cyl_radius)
                    {
                        mask[ii] = SOLID;
                    }
                    else
                    {
                        mask[ii] = FLUID;

                        double dist2 = (x - pulse_center_x) * (x - pulse_center_x) +
                                       (y - pulse_center_y) * (y - pulse_center_y);
                        double rho_prime = pulse_amp_lbm * std::exp(-std::log(2.0) * dist2 / (pulse_width_b * pulse_width_b));
                        rho[ii] = 1.0 + rho_prime;

                        for (int q = 0; q < Q; ++q)
                        {
                            double feq = get_feq(q, rho[ii], 0.0, 0.0, 0.0);
                            f_curr[ii * Q + q] = feq;
                            f_next[ii * Q + q] = feq;
                        }

                        double dist_x = 0.0;
                        double dist_y = 0.0;

                        if (x < X_MIN + sponge_thickness) dist_x = (X_MIN + sponge_thickness) - x;
                        else if (x > X_MAX - sponge_thickness) dist_x = x - (X_MAX - sponge_thickness);

                        if (y < Y_MIN + sponge_thickness) dist_y = (Y_MIN + sponge_thickness) - y;
                        else if (y > Y_MAX - sponge_thickness) dist_y = y - (Y_MAX - sponge_thickness);

                        double dist_max = std::max(dist_x, dist_y);

                        if (dist_max > 0.0)
                        {
                            double r_ratio = dist_max / sponge_thickness;
                            sponge_sigma[ii] = damping_strength * r_ratio * r_ratio;
                        }
                        else
                        {
                            sponge_sigma[ii] = 0.0;
                        }
                    }
                }
    }

    void exchange_halos()
    {
        int count_x = ny * nz * Q;
        int count_y = (nx + 2) * nz * Q;
        int count_z = (nx + 2) * (ny + 2) * Q;

        std::vector<double> sendL(count_x), recvL(count_x), sendR(count_x), recvR(count_x);
        std::vector<double> sendF(count_y), recvF(count_y), sendB(count_y), recvB(count_y);
        std::vector<double> sendD(count_z), recvD(count_z), sendU(count_z), recvU(count_z);

        double feq0[Q];
        for (int q = 0; q < Q; ++q) feq0[q] = get_feq(q, 1.0, 0, 0, 0);
        auto fill = [&](std::vector<double> &v)
        {
            for (size_t i = 0; i < v.size() / Q; ++i) for (int q = 0; q < Q; ++q) v[i * Q + q] = feq0[q];
        };
        fill(recvL);
        fill(recvR);
        fill(recvF);
        fill(recvB);
        fill(recvD);
        fill(recvU);

        int cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j) for (int q = 0; q < Q; ++q) sendL[cnt++] = f_curr[loc_index(1, j, k, nx_t, ny_t) * Q + q];
        cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j) for (int q = 0; q < Q; ++q) sendR[cnt++] = f_curr[loc_index(nx, j, k, nx_t, ny_t) * Q + q];

        MPI_Request reqs[4];
        MPI_Irecv(recvL.data(), count_x, MPI_DOUBLE, nbr[0], 10, cart, &reqs[0]);
        MPI_Irecv(recvR.data(), count_x, MPI_DOUBLE, nbr[1], 11, cart, &reqs[1]);
        MPI_Isend(sendR.data(), count_x, MPI_DOUBLE, nbr[1], 10, cart, &reqs[2]);
        MPI_Isend(sendL.data(), count_x, MPI_DOUBLE, nbr[0], 11, cart, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j) for (int q = 0; q < Q; ++q) f_curr[loc_index(0, j, k, nx_t, ny_t) * Q + q] = recvL[cnt++];
        cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j) for (int q = 0; q < Q; ++q) f_curr[loc_index(nx + 1, j, k, nx_t, ny_t) * Q + q] = recvR[cnt++];

        cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i) for (int q = 0; q < Q; ++q) sendF[cnt++] = f_curr[loc_index(i, 1, k, nx_t, ny_t) * Q + q];
        cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i) for (int q = 0; q < Q; ++q) sendB[cnt++] = f_curr[loc_index(i, ny, k, nx_t, ny_t) * Q + q];

        MPI_Irecv(recvF.data(), count_y, MPI_DOUBLE, nbr[2], 20, cart, &reqs[0]);
        MPI_Irecv(recvB.data(), count_y, MPI_DOUBLE, nbr[3], 21, cart, &reqs[1]);
        MPI_Isend(sendB.data(), count_y, MPI_DOUBLE, nbr[3], 20, cart, &reqs[2]);
        MPI_Isend(sendF.data(), count_y, MPI_DOUBLE, nbr[2], 21, cart, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i) for (int q = 0; q < Q; ++q) f_curr[loc_index(i, 0, k, nx_t, ny_t) * Q + q] = recvF[cnt++];
        cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i) for (int q = 0; q < Q; ++q) f_curr[loc_index(i, ny + 1, k, nx_t, ny_t) * Q + q] = recvB[cnt++];

        cnt = 0;
        for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i) for (int q = 0; q < Q; ++q) sendD[cnt++] = f_curr[loc_index(i, j, 1, nx_t, ny_t) * Q + q];
        cnt = 0;
        for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i) for (int q = 0; q < Q; ++q) sendU[cnt++] = f_curr[loc_index(i, j, nz, nx_t, ny_t) * Q + q];

        MPI_Irecv(recvD.data(), count_z, MPI_DOUBLE, nbr[4], 30, cart, &reqs[0]);
        MPI_Irecv(recvU.data(), count_z, MPI_DOUBLE, nbr[5], 31, cart, &reqs[1]);
        MPI_Isend(sendU.data(), count_z, MPI_DOUBLE, nbr[5], 30, cart, &reqs[2]);
        MPI_Isend(sendD.data(), count_z, MPI_DOUBLE, nbr[4], 31, cart, &reqs[3]);
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        cnt = 0;
        for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i) for (int q = 0; q < Q; ++q) f_curr[loc_index(i, j, 0, nx_t, ny_t) * Q + q] = recvD[cnt++];
        cnt = 0;
        for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i) for (int q = 0; q < Q; ++q) f_curr[loc_index(i, j, nz + 1, nx_t, ny_t) * Q + q] = recvU[cnt++];
    }

    void step_pull()
    {
        double tau_fluid = 0.505;

        #pragma omp parallel for collapse(3)
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j) for (int i = 1; i <= nx; ++i)
                {
                    size_t dst = loc_index(i, j, k, nx_t, ny_t);
                    if (mask[dst] == SOLID) continue;

                    double sigma = sponge_sigma[dst];
                    bool in_sponge = (sigma > 1e-6);

                    for (int q = 0; q < Q; ++q)
                    {
                        size_t src = loc_index(i - ex[q], j - ey[q], k - ez[q], nx_t, ny_t);
                        if (mask[src] == SOLID)
                        {
                            f_next[dst * Q + q] = f_curr[dst * Q + inv_q[q]];
                        }
                        else
                        {
                            double sr = 0.0, sux = 0.0, suy = 0.0, suz = 0.0;
                            size_t src_base = src * Q;
                            for (int p = 0; p < Q; ++p)
                            {
                                double val = f_curr[src_base + p];
                                sr += val;
                                sux += val * ex[p];
                                suy += val * ey[p];
                                suz += val * ez[p];
                            }
                            if (sr == 0.0) sr = 1.0;
                            sux /= sr;
                            suy /= sr;
                            suz /= sr;

                            double feq = get_feq(q, sr, sux, suy, suz);
                            double f_post = f_curr[src_base + q] - (f_curr[src_base + q] - feq) / tau_fluid;

                            if (in_sponge)
                            {
                                double feq_rest = get_feq(q, 1.0, 0.0, 0.0, 0.0);
                                f_next[dst * Q + q] = (1.0 - sigma) * f_post + sigma * feq_rest;
                            }
                            else
                            {
                                f_next[dst * Q + q] = f_post;
                            }
                        }
                    }
                    double sumr = 0.0;
                    for (int p = 0; p < Q; ++p) sumr += f_next[dst * Q + p];
                    rho[dst] = sumr;
                }
        f_curr.swap(f_next);
    }

    void write_vtk(int step)
    {
        std::vector<double> local_rho((size_t)nx * ny * nz);
        int idxl = 0;
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j) for (int i = 1; i <= nx; ++i)
                {
                    size_t li = loc_index(i, j, k, nx_t, ny_t);
                    if (mask[li] == SOLID) local_rho[idxl++] = 1.0;
                    else local_rho[idxl++] = rho[li];
                }

        if (rank == 0)
        {
            size_t total_points = (size_t)NXg * NYg * NZg;
            std::vector<double> global_rho(total_points, 1.0);

            idxl = 0;
            for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i)
                    {
                        size_t gi = (size_t)(off_z + k) * NXg * NYg + (size_t)(off_y + j) * NXg + (size_t)(off_x + i);
                        if (gi < total_points) global_rho[gi] = local_rho[idxl++];
                    }

            for (int p = 1; p < size; ++p)
            {
                int coords_p[3];
                MPI_Cart_coords(cart, p, 3, coords_p);
                auto block_low = [&](int id, int pnum, int n)
                {
                    return id * n / pnum;
                };
                auto block_size = [&](int id, int pnum, int n)
                {
                    return block_low(id + 1, pnum, n) - block_low(id, pnum, n);
                };
                int offxp = block_low(coords_p[0], dims[0], NXg);
                int offyp = block_low(coords_p[1], dims[1], NYg);
                int offzp = block_low(coords_p[2], dims[2], NZg);
                int nxp = block_size(coords_p[0], dims[0], NXg);
                int nyp = block_size(coords_p[1], dims[1], NYg);
                int nzp = block_size(coords_p[2], dims[2], NZg);

                std::vector<double> buf((size_t)nxp * nyp * nzp);
                MPI_Recv(buf.data(), (int)buf.size(), MPI_DOUBLE, p, 999, cart, MPI_STATUS_IGNORE);

                int idr = 0;
                for (int k = 0; k < nzp; ++k) for (int j = 0; j < nyp; ++j) for (int i = 0; i < nxp; ++i)
                        {
                            size_t gi = (size_t)(offzp + k) * NXg * NYg + (size_t)(offyp + j) * NXg + (size_t)(offxp + i);
                            if (gi < total_points) global_rho[gi] = buf[idr++];
                        }
            }

            std::ostringstream fn;
            fn << "output_mpi/cyl_" << std::setw(5) << std::setfill('0') << step << ".vtk";
            std::ofstream ofs(fn.str());

            ofs << "# vtk DataFile Version 3.0\n";
            ofs << "LBM Final Corrected\n";
            ofs << "ASCII\n";
            ofs << "DATASET STRUCTURED_POINTS\n";
            ofs << "DIMENSIONS " << NXg << " " << NYg << " " << NZg << "\n";
            ofs << "ORIGIN " << X_MIN << " " << Y_MIN << " " << Z_MIN << "\n";
            ofs << "SPACING " << units.dx_phys << " " << units.dx_phys << " " << units.dx_phys << "\n";
            ofs << "POINT_DATA " << total_points << "\n";
            ofs << "SCALARS Pressure double 1\n";
            ofs << "LOOKUP_TABLE default\n";

            ofs << std::scientific << std::setprecision(6);
            for (size_t v = 0; v < total_points; ++v)
            {
                double val = global_rho[v];
                if (std::isnan(val) || std::isinf(val)) val = 1.0;

                double p_val = (val - 1.0) * output_scale_factor;
                ofs << p_val << "\n";
            }
            ofs.flush();
            ofs.close();
        }
        else
        {
            MPI_Send(local_rho.data(), (int)local_rho.size(), MPI_DOUBLE, 0, 999, cart);
        }
        MPI_Barrier(cart);
    }

    void record_probes(double time)
    {
        for (auto& p : probes)
        {
            int gx_idx = (int)((p.x - X_MIN) / units.dx_phys - 0.5);
            int gy_idx = (int)((p.y - Y_MIN) / units.dx_phys - 0.5);
            int gz_idx = (int)((p.z - Z_MIN) / units.dx_phys - 0.5);

            int lx = gx_idx - off_x + 1;
            int ly = gy_idx - off_y + 1;
            int lz = gz_idx - off_z + 1;

            double pressure_val = 0.0;
            int found = 0;

            if (lx >= 1 && lx <= nx && ly >= 1 && ly <= ny && lz >= 1 && lz <= nz)
            {
                size_t idx = loc_index(lx, ly, lz, nx_t, ny_t);
                pressure_val = (rho[idx] - 1.0) * output_scale_factor;
                found = 1;
            }
            double global_val = 0.0;
            int global_found = 0;
            MPI_Reduce(&pressure_val, &global_val, 1, MPI_DOUBLE, MPI_SUM, 0, cart);
            MPI_Reduce(&found, &global_found, 1, MPI_INT, MPI_SUM, 0, cart);

            if (rank == 0 && global_found > 0)
            {
                p.history.push_back({time, global_val});
            }
        }
    }

    void write_probes_to_file()
    {
        if (rank != 0) return;
        for (const auto& p : probes)
        {
            std::string fname = "output_mpi/probe_" + p.name + ".dat";
            std::ofstream ofs(fname);
            ofs << "# Time Pressure\n";
            for (const auto& h : p.history)
            {
                ofs << h.first << " " << h.second << "\n";
            }
            ofs.close();
        }
    }

    void run_simulation()
    {
        double dt_phys = units.get_dt_phys();
        double T_max = 11.0;
        int max_steps = (int)(T_max / dt_phys) + 20;

        std::vector<double> vtk_times = {3.0, 6.0};
        auto next_vtk = vtk_times.begin();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int s = 0; s <= max_steps; ++s)
        {
            double time = s * dt_phys;
            record_probes(time);

            bool dump = false;
            if (next_vtk != vtk_times.end() && time >= *next_vtk)
            {
                dump = true;
                next_vtk++;
            }
            if (s == 0) dump = true;

            if (dump)
            {
                if (rank == 0) std::cout << "Output at step " << s << " t=" << time << "\n";
                write_vtk(s);
            }

            exchange_halos();
            step_pull();
        }
        write_probes_to_file();
        auto t1 = std::chrono::high_resolution_clock::now();
        t_run = std::chrono::duration<double>(t1 - t0).count();
        if (rank == 0) std::cout << "Run finished in " << t_run << " s\n";
    }
};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0) std::filesystem::create_directories("output_mpi");
    Solver solver;
    solver.setup_mpi(argc, argv);
    solver.init_physics();
    solver.run_simulation();
    MPI_Finalize();
    return 0;
}