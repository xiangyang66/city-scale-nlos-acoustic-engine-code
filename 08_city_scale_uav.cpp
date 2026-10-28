#include <mpi.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static constexpr int Q = 19;
static constexpr double cs2 = 1.0 / 3.0;
static constexpr double cs = 0.57735026918962576451;
static constexpr double PI = 3.14159265358979323846;

static constexpr int ex[Q] = {0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1, 0, 0, 0, 0};
static constexpr int ey[Q] = {0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1};
static constexpr int ez[Q] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1};

static constexpr double w[Q] =
{
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

static int inv_q[Q];
static int q_of_v[3][3][3];

enum CellType : uint8_t { FLUID = 0, SOLID = 1 };

struct Point
{
    double x, y, z;
};

struct Triangle
{
    Point v1, v2, v3;
    Point minPt, maxPt;
};

#pragma pack(push, 1)
struct StlTriangle
{
    float n[3];
    float v1[3];
    float v2[3];
    float v3[3];
    uint16_t attr;
};
#pragma pack(pop)

class GeometryEngine
{
public:
    std::vector<Triangle> triangles;
    Point geoMin{1e30, 1e30, 1e30}, geoMax{-1e30, -1e30, -1e30};

    int gridRows = 200, gridCols = 200;
    double cellW = 1.0, cellH = 1.0;
    std::vector<std::vector<int>> grid;

    bool load(const std::string& filename, double scale, double sx, double sy, double sz)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;

        char header[80];
        file.read(header, 80);
        uint32_t numTri = 0;
        file.read(reinterpret_cast<char*>(&numTri), 4);

        triangles.resize(numTri);
        geoMin = {1e30, 1e30, 1e30};
        geoMax = {-1e30, -1e30, -1e30};

        for (uint32_t i = 0; i < numTri; ++i)
        {
            StlTriangle st{};
            file.read(reinterpret_cast<char*>(&st), 50);

            Triangle& t = triangles[i];
            t.v1 = {st.v1[0] *scale + sx, st.v1[1] *scale + sy, st.v1[2] *scale + sz};
            t.v2 = {st.v2[0] *scale + sx, st.v2[1] *scale + sy, st.v2[2] *scale + sz};
            t.v3 = {st.v3[0] *scale + sx, st.v3[1] *scale + sy, st.v3[2] *scale + sz};

            t.minPt.x = std::min({t.v1.x, t.v2.x, t.v3.x});
            t.minPt.y = std::min({t.v1.y, t.v2.y, t.v3.y});
            t.minPt.z = std::min({t.v1.z, t.v2.z, t.v3.z});
            t.maxPt.x = std::max({t.v1.x, t.v2.x, t.v3.x});
            t.maxPt.y = std::max({t.v1.y, t.v2.y, t.v3.y});
            t.maxPt.z = std::max({t.v1.z, t.v2.z, t.v3.z});

            geoMin.x = std::min(geoMin.x, t.minPt.x);
            geoMin.y = std::min(geoMin.y, t.minPt.y);
            geoMin.z = std::min(geoMin.z, t.minPt.z);
            geoMax.x = std::max(geoMax.x, t.maxPt.x);
            geoMax.y = std::max(geoMax.y, t.maxPt.y);
            geoMax.z = std::max(geoMax.z, t.maxPt.z);
        }

        buildGrid();
        return true;
    }

    void buildGrid()
    {
        if (triangles.empty()) return;

        cellW = (geoMax.x - geoMin.x) / gridRows;
        cellH = (geoMax.y - geoMin.y) / gridCols;
        if (cellW < 1e-9) cellW = 1.0;
        if (cellH < 1e-9) cellH = 1.0;

        grid.assign(gridRows * gridCols, {});

        for (int i = 0; i < (int)triangles.size(); ++i)
        {
            const auto& t = triangles[i];
            int r1 = (int)((t.minPt.x - geoMin.x) / cellW);
            int r2 = (int)((t.maxPt.x - geoMin.x) / cellW);
            int c1 = (int)((t.minPt.y - geoMin.y) / cellH);
            int c2 = (int)((t.maxPt.y - geoMin.y) / cellH);

            r1 = std::max(0, std::min(gridRows - 1, r1));
            r2 = std::max(0, std::min(gridRows - 1, r2));
            c1 = std::max(0, std::min(gridCols - 1, c1));
            c2 = std::max(0, std::min(gridCols - 1, c2));

            for (int r = r1; r <= r2; ++r) for (int c = c1; c <= c2; ++c)
                {
                    grid[r * gridCols + c].push_back(i);
                }
        }
    }

    double getGroundHeight(double x, double y) const
    {
        if (triangles.empty()) return -1e30;
        if (x < geoMin.x || x > geoMax.x || y < geoMin.y || y > geoMax.y) return -1e30;

        int r = (int)((x - geoMin.x) / cellW);
        int c = (int)((y - geoMin.y) / cellH);
        if (r < 0 || r >= gridRows || c < 0 || c >= gridCols) return -1e30;

        const auto& candidates = grid[r * gridCols + c];
        if (candidates.empty()) return -1e30;

        const Point ro{x, y, 10000.0};
        const Point rd{0.0, 0.0, -1.0};

        double maxZ = -1e30;
        for (int idx : candidates)
        {
            const auto& tri = triangles[idx];

            if (tri.maxPt.x < x - 1e-6 || tri.minPt.x > x + 1e-6) continue;
            if (tri.maxPt.y < y - 1e-6 || tri.minPt.y > y + 1e-6) continue;

            const double EPS = 1e-9;

            Point e1{tri.v2.x - tri.v1.x, tri.v2.y - tri.v1.y, tri.v2.z - tri.v1.z};
            Point e2{tri.v3.x - tri.v1.x, tri.v3.y - tri.v1.y, tri.v3.z - tri.v1.z};
            Point h{rd.y *e2.z - rd.z * e2.y, rd.z *e2.x - rd.x * e2.z, rd.x *e2.y - rd.y * e2.x};
            double a = e1.x * h.x + e1.y * h.y + e1.z * h.z;
            if (std::abs(a) < EPS) continue;

            double f = 1.0 / a;
            Point s{ro.x - tri.v1.x, ro.y - tri.v1.y, ro.z - tri.v1.z};
            double u = f * (s.x * h.x + s.y * h.y + s.z * h.z);
            if (u < 0.0 || u > 1.0) continue;

            Point q{s.y *e1.z - s.z * e1.y, s.z *e1.x - s.x * e1.z, s.x *e1.y - s.y * e1.x};
            double v = f * (rd.x * q.x + rd.y * q.y + rd.z * q.z);
            if (v < 0.0 || (u + v) > 1.0) continue;

            double t = f * (e2.x * q.x + e2.y * q.y + e2.z * q.z);
            if (t > EPS)
            {
                double hitZ = ro.z + t * rd.z;
                if (hitZ > maxZ) maxZ = hitZ;
            }
        }
        return maxZ;
    }
};

struct UnitConverter
{
    double dx_phys = 2.0;
    double c_phys  = 340.0;

    double get_dt_phys() const
    {
        return (cs * dx_phys) / c_phys;
    }
};

static inline size_t loc_index(int i, int j, int k, int nx_t, int ny_t)
{
    return (size_t)i + (size_t)j * (size_t)nx_t + (size_t)k * (size_t)nx_t * (size_t)ny_t;
}

struct DroneSource
{
    double sx, sy, sz;
    double ex, ey, ez;
    double speed;
    double amp;
    double freq;

    double cx, cy, cz;

    void update(double t)
    {
        double dx = ex - sx, dy = ey - sy, dz = ez - sz;
        double dist_total = std::sqrt(dx * dx + dy * dy + dz * dz);
        if (dist_total < 1e-12 || speed < 1e-12)
        {
            cx = sx;
            cy = sy;
            cz = sz;
            return;
        }
        double travel_time = dist_total / speed;
        double r = t / travel_time;
        if (r > 1.0) r = 1.0;
        cx = sx + r * dx;
        cy = sy + r * dy;
        cz = sz + r * dz;
    }
};

class LBMSolver
{
public:
    std::string stl_file;
    double stl_sx = 0.0, stl_sy = 0.0, stl_sz = 0.0;
    DroneSource drone;

    bool use_geometry = true;

    double X_MIN = 0.0, X_MAX = 1650.0;
    double Y_MIN = 0.0, Y_MAX = 1150.0;
    double Z_MIN = 0.0, Z_MAX = 250.0;
    double DX = 2.0;

    bool output_world_coords = true;

    double tau = 0.55;

private:
    MPI_Comm cart = MPI_COMM_NULL;
    int rank = 0, size = 1;
    int dims[3] {0, 0, 0}, coords[3] {0, 0, 0}, nbr[6] {MPI_PROC_NULL};
    int NXg = 0, NYg = 0, NZg = 0;
    int nx = 0, ny = 0, nz = 0, off_x = 0, off_y = 0, off_z = 0;
    int nx_t = 0, ny_t = 0, nz_t = 0;

    UnitConverter units;
    GeometryEngine geo;

    std::vector<double> f_curr, f_next, rho, sponge_sigma;
    std::vector<uint8_t> mask;

    std::vector<double> sL, rL, sR, rR, sF, rF, sB, rB, sD, rD, sU, rU;

    inline double feq(int k, double r, double ux, double uy, double uz) const
    {
        double cu = ex[k] * ux + ey[k] * uy + ez[k] * uz;
        double u2 = ux * ux + uy * uy + uz * uz;
        return w[k] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }

    std::array<double, 3> world_offset() const
    {
        return {-stl_sx, -stl_sy, -stl_sz};
    }

public:
    void init(int argc, char** argv)
    {
        for (int i = 0; i < Q; ++i)
        {
            inv_q[i] = 0;
            for (int j = 0; j < Q; ++j)
            {
                if (ex[i] == -ex[j] && ey[i] == -ey[j] && ez[i] == -ez[j])
                {
                    inv_q[i] = j;
                    break;
                }
            }
        }

        for (int ix = 0; ix < 3; ++ix)
            for (int iy = 0; iy < 3; ++iy)
                for (int iz = 0; iz < 3; ++iz)
                    q_of_v[ix][iy][iz] = -1;
        for (int q = 0; q < Q; ++q)
        {
            const int ix = ex[q] + 1;
            const int iy = ey[q] + 1;
            const int iz = ez[q] + 1;
            if (ix >= 0 && ix < 3 && iy >= 0 && iy < 3 && iz >= 0 && iz < 3)
                q_of_v[ix][iy][iz] = q;
        }
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        units.dx_phys = DX;

        NXg = (int)((X_MAX - X_MIN) / units.dx_phys + 0.5);
        NYg = (int)((Y_MAX - Y_MIN) / units.dx_phys + 0.5);
        NZg = (int)((Z_MAX - Z_MIN) / units.dx_phys + 0.5);

        dims[0] = dims[1] = dims[2] = 0;
        MPI_Dims_create(size, 3, dims);
        int periods[3] = {0, 0, 0};
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
        size_t Nloc = (size_t)nx_t * (size_t)ny_t * (size_t)nz_t;

        f_curr.assign(Nloc * Q, 0.0);
        f_next.assign(Nloc * Q, 0.0);
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
        for (int q = 0; q < Q; ++q) feq0[q] = feq(q, 1.0, 0.0, 0.0, 0.0);

        #pragma omp parallel for
        for (size_t i = 0; i < Nloc; ++i)
        {
            for (int q = 0; q < Q; ++q) f_curr[i * Q + q] = feq0[q];
        }

        bool ok = true;
        if (use_geometry)
        {
            ok = geo.load(stl_file, 1.0, stl_sx, stl_sy, stl_sz);
        }
        else
        {
            ok = true;
        }

        if (rank == 0)
        {
            if (!ok) std::cout << "Warning: STL not loaded, using empty geometry.\n";
            else     std::cout << "STL loaded: triangles=" << geo.triangles.size() << "\n";
        }

        init_domain();

        if (rank == 0)
        {
            std::cout << "\n--- V21 D3Q19-BGK Collision + World VTK ---\n";
            std::cout << "MPI dims: " << dims[0] << " x " << dims[1] << " x " << dims[2] << "\n";
            std::cout << "Grid: " << NXg << " x " << NYg << " x " << NZg << "  (dx=" << units.dx_phys << " m)\n";
            std::cout << "dt=" << units.get_dt_phys() << " s\n";
            std::cout << "Drone path: (" << drone.sx << "," << drone.sy << "," << drone.sz << ") -> ("
                      << drone.ex << "," << drone.ey << "," << drone.ez << "), speed=" << drone.speed << " m/s\n";
            auto off = world_offset();
            std::cout << "World offset (added to VTK origin): (" << off[0] << "," << off[1] << "," << off[2] << ")\n\n";
        }
    }

private:
    void init_domain()
    {
        const double sponge_w = 20.0 * units.dx_phys;
        const double sponge_strength = 0.35;

        const double sky_z = 120.0;

        std::vector<double> gH_xy((size_t)nx_t * (size_t)ny_t, -1e30);
        #pragma omp parallel for collapse(2)
        for (int j = 0; j < ny_t; ++j) for (int i = 0; i < nx_t; ++i)
            {
                double x = X_MIN + (off_x + (i - 1) + 0.5) * units.dx_phys;
                double y = Y_MIN + (off_y + (j - 1) + 0.5) * units.dx_phys;
                gH_xy[(size_t)j * nx_t + (size_t)i] = geo.getGroundHeight(x, y);
            }

        #pragma omp parallel for collapse(3)
        for (int k = 0; k < nz_t; ++k) for (int j = 0; j < ny_t; ++j) for (int i = 0; i < nx_t; ++i)
                {
                    size_t idx = loc_index(i, j, k, nx_t, ny_t);

                    double x = X_MIN + (off_x + (i - 1) + 0.5) * units.dx_phys;
                    double y = Y_MIN + (off_y + (j - 1) + 0.5) * units.dx_phys;
                    double z = Z_MIN + (off_z + (k - 1) + 0.5) * units.dx_phys;

                    double gH = gH_xy[(size_t)j * nx_t + (size_t)i];

                    if (!use_geometry)
                    {
                        mask[idx] = FLUID;
                    }
                    else
                    {
                        mask[idx] = (z < gH) ? SOLID : FLUID;
                    }

                    if (z > sky_z) mask[idx] = FLUID;

                    double d = 1e30;
                    d = std::min(d, x - X_MIN);
                    d = std::min(d, X_MAX - x);
                    d = std::min(d, y - Y_MIN);
                    d = std::min(d, Y_MAX - y);
                    d = std::min(d, Z_MAX - z);

                    if (d < sponge_w)
                    {
                        double r = (sponge_w - d) / sponge_w;
                        sponge_sigma[idx] = sponge_strength * r * r;
                    }
                    else
                    {
                        sponge_sigma[idx] = 0.0;
                    }
                }

        for (int k = 0; k <= nz + 1; ++k) for (int j = 0; j <= ny + 1; ++j)
            {
                mask[loc_index(0,      j, k, nx_t, ny_t)] = FLUID;
                mask[loc_index(nx + 1, j, k, nx_t, ny_t)] = FLUID;
            }
        for (int k = 0; k <= nz + 1; ++k) for (int i = 0; i <= nx + 1; ++i)
            {
                mask[loc_index(i, 0,      k, nx_t, ny_t)] = FLUID;
                mask[loc_index(i, ny + 1, k, nx_t, ny_t)] = FLUID;
            }

        for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i)
            {
                if (nbr[4] == MPI_PROC_NULL)
                {
                    mask[loc_index(i, j, 0,      nx_t, ny_t)] = SOLID;
                }
                else
                {
                    mask[loc_index(i, j, 0,      nx_t, ny_t)] = FLUID;
                }
                mask[loc_index(i, j, nz + 1, nx_t, ny_t)] = FLUID;
            }
    }

    void fill_with_feq0(std::vector<double> &buf) const
    {
        double feq0[Q];
        for (int q = 0; q < Q; ++q) feq0[q] = feq(q, 1.0, 0.0, 0.0, 0.0);
        size_t cells = buf.size() / Q;
        for (size_t c = 0; c < cells; ++c)
        {
            for (int q = 0; q < Q; ++q) buf[c * Q + q] = feq0[q];
        }
    }

    inline void macro_at(size_t idx, double &r, double &ux, double &uy, double &uz) const
    {
        r = ux = uy = uz = 0.0;
        const size_t base = idx * Q;
        for (int q = 0; q < Q; ++q)
        {
            const double fq = f_curr[base + q];
            r  += fq;
            ux += fq * ex[q];
            uy += fq * ey[q];
            uz += fq * ez[q];
        }
        if (r > 1e-30)
        {
            ux /= r;
            uy /= r;
            uz /= r;
        }
        else
        {
            r = 1.0;
            ux = uy = uz = 0.0;
        }
    }

    inline void feq_array(double r, double ux, double uy, double uz, double out[Q]) const
    {
        for (int q = 0; q < Q; ++q) out[q] = feq(q, r, ux, uy, uz);
    }

public:
    void apply_source(double t)
    {
        const double dt  = units.get_dt_phys();
        const double val = drone.amp * std::sin(2.0 * PI * drone.freq * t);
        const double source_gain = 1.0;

        drone.update(t);

        const int    R = 2;
        const double sigma_cells = 1.25;
        const double sigma2 = (sigma_cells * DX) * (sigma_cells * DX);

        const int ic = (int)std::floor((drone.cx - X_MIN) / DX);
        const int jc = (int)std::floor((drone.cy - Y_MIN) / DX);
        const int kc = (int)std::floor((drone.cz - Z_MIN) / DX);

        if (ic < -R || ic >= NXg + R || jc < -R || jc >= NYg + R || kc < -R || kc >= NZg + R) return;

        double sumw_local = 0.0;

        for (int gk = kc - R; gk <= kc + R; ++gk)
            for (int gj = jc - R; gj <= jc + R; ++gj)
                for (int gi = ic - R; gi <= ic + R; ++gi)
                {
                    if (gi < 0 || gi >= NXg || gj < 0 || gj >= NYg || gk < 0 || gk >= NZg) continue;

                    const double x = X_MIN + (gi + 0.5) * DX;
                    const double y = Y_MIN + (gj + 0.5) * DX;
                    const double z = Z_MIN + (gk + 0.5) * DX;

                    const double dx = x - drone.cx;
                    const double dy = y - drone.cy;
                    const double dz = z - drone.cz;
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    const double ww = std::exp(-0.5 * r2 / sigma2);

                    if (gi < off_x || gi >= off_x + nx) continue;
                    if (gj < off_y || gj >= off_y + ny) continue;
                    if (gk < off_z || gk >= off_z + nz) continue;

                    const int li = gi - off_x + 1;
                    const int lj = gj - off_y + 1;
                    const int lk = gk - off_z + 1;
                    const size_t idx = loc_index(li, lj, lk, nx_t, ny_t);

                    if (mask[idx] == FLUID) sumw_local += ww;
                }

        double sumw_global = 0.0;
        MPI_Allreduce(&sumw_local, &sumw_global, 1, MPI_DOUBLE, MPI_SUM, cart);
        if (sumw_global < 1e-30) return;

        const double A = (val * dt * source_gain) / sumw_global;

        for (int gk = kc - R; gk <= kc + R; ++gk)
            for (int gj = jc - R; gj <= jc + R; ++gj)
                for (int gi = ic - R; gi <= ic + R; ++gi)
                {
                    if (gi < 0 || gi >= NXg || gj < 0 || gj >= NYg || gk < 0 || gk >= NZg) continue;

                    if (gi < off_x || gi >= off_x + nx) continue;
                    if (gj < off_y || gj >= off_y + ny) continue;
                    if (gk < off_z || gk >= off_z + nz) continue;

                    const double x = X_MIN + (gi + 0.5) * DX;
                    const double y = Y_MIN + (gj + 0.5) * DX;
                    const double z = Z_MIN + (gk + 0.5) * DX;

                    const double dx = x - drone.cx;
                    const double dy = y - drone.cy;
                    const double dz = z - drone.cz;
                    const double r2 = dx * dx + dy * dy + dz * dz;
                    const double ww = std::exp(-0.5 * r2 / sigma2);

                    const int li = gi - off_x + 1;
                    const int lj = gj - off_y + 1;
                    const int lk = gk - off_z + 1;
                    const size_t idx = loc_index(li, lj, lk, nx_t, ny_t);

                    if (mask[idx] != FLUID) continue;

                    const double add  = A * ww;
                    const size_t base = idx * Q;
                    for (int q = 0; q < Q; ++q)
                    {
                        f_curr[base + q] += w[q] * add;
                    }
                }
    }

    bool use_nscbc = true;

    void exchange_halos()
    {
        const int cx = ny * nz * Q;
        const int cy = (nx + 2) * nz * Q;
        const int cz = (nx + 2) * (ny + 2) * Q;

        fill_with_feq0(rL);
        fill_with_feq0(rR);
        fill_with_feq0(rF);
        fill_with_feq0(rB);
        fill_with_feq0(rD);
        fill_with_feq0(rU);

        if (use_nscbc && nbr[0] == MPI_PROC_NULL)
        {
            int cnt = 0;
            for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j)
                {
                    const size_t idxI = loc_index(1, j, k, nx_t, ny_t);
                    double rI, uxI, uyI, uzI;
                    macro_at(idxI, rI, uxI, uyI, uzI);
                    const double pI = cs2 * (rI - 1.0);

                    const double Wp = 0.0;
                    const double Wm = uxI - pI / cs;

                    const double uxB = 0.5 * (Wp + Wm);
                    const double pB  = 0.5 * cs * (Wp - Wm);
                    const double rhoB  = 1.0 + pB / cs2;

                    double feqI[Q], feqB[Q];
                    feq_array(rI, uxI, uyI, uzI, feqI);
                    feq_array(rhoB, uxB, uyI, uzI, feqB);

                    const size_t baseI = idxI * Q;
                    for (int q = 0; q < Q; ++q)
                        rL[cnt++] = feqB[q] + (f_curr[baseI + q] - feqI[q]);
                }
        }

        if (use_nscbc && nbr[1] == MPI_PROC_NULL)
        {
            int cnt = 0;
            for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j)
                {
                    const size_t idxI = loc_index(nx, j, k, nx_t, ny_t);
                    double rI, uxI, uyI, uzI;
                    macro_at(idxI, rI, uxI, uyI, uzI);
                    const double pI = cs2 * (rI - 1.0);

                    const double Wm = 0.0;
                    const double Wp = uxI + pI / cs;

                    const double uxB = 0.5 * (Wp + Wm);
                    const double pB  = 0.5 * cs * (Wp - Wm);
                    const double rhoB  = 1.0 + pB / cs2;

                    double feqI[Q], feqB[Q];
                    feq_array(rI, uxI, uyI, uzI, feqI);
                    feq_array(rhoB, uxB, uyI, uzI, feqB);

                    const size_t baseI = idxI * Q;
                    for (int q = 0; q < Q; ++q)
                        rR[cnt++] = feqB[q] + (f_curr[baseI + q] - feqI[q]);
                }
        }

        if (use_nscbc && nbr[2] == MPI_PROC_NULL)
        {
            int cnt = 0;
            for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i)
                {
                    if (i == 0 || i == nx + 1)
                    {
                        cnt += Q;
                        continue;
                    }

                    const size_t idxI = loc_index(i, 1, k, nx_t, ny_t);
                    double rI, uxI, uyI, uzI;
                    macro_at(idxI, rI, uxI, uyI, uzI);
                    const double pI = cs2 * (rI - 1.0);

                    const double Wp = 0.0;
                    const double Wm = uyI - pI / cs;

                    const double uyB = 0.5 * (Wp + Wm);
                    const double pB  = 0.5 * cs * (Wp - Wm);
                    const double rhoB  = 1.0 + pB / cs2;

                    double feqI[Q], feqB[Q];
                    feq_array(rI, uxI, uyI, uzI, feqI);
                    feq_array(rhoB, uxI, uyB, uzI, feqB);

                    const size_t baseI = idxI * Q;
                    for (int q = 0; q < Q; ++q)
                        rF[cnt++] = feqB[q] + (f_curr[baseI + q] - feqI[q]);
                }
        }

        if (use_nscbc && nbr[3] == MPI_PROC_NULL)
        {
            int cnt = 0;
            for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i)
                {
                    if (i == 0 || i == nx + 1)
                    {
                        cnt += Q;
                        continue;
                    }

                    const size_t idxI = loc_index(i, ny, k, nx_t, ny_t);
                    double rI, uxI, uyI, uzI;
                    macro_at(idxI, rI, uxI, uyI, uzI);
                    const double pI = cs2 * (rI - 1.0);

                    const double Wm = 0.0;
                    const double Wp = uyI + pI / cs;

                    const double uyB = 0.5 * (Wp + Wm);
                    const double pB  = 0.5 * cs * (Wp - Wm);
                    const double rhoB  = 1.0 + pB / cs2;

                    double feqI[Q], feqB[Q];
                    feq_array(rI, uxI, uyI, uzI, feqI);
                    feq_array(rhoB, uxI, uyB, uzI, feqB);

                    const size_t baseI = idxI * Q;
                    for (int q = 0; q < Q; ++q)
                        rB[cnt++] = feqB[q] + (f_curr[baseI + q] - feqI[q]);
                }
        }

        if (use_nscbc && nbr[5] == MPI_PROC_NULL)
        {
            int cnt = 0;
            for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i)
                {
                    if (i == 0 || i == nx + 1 || j == 0 || j == ny + 1)
                    {
                        cnt += Q;
                        continue;
                    }

                    const size_t idxI = loc_index(i, j, nz, nx_t, ny_t);
                    double rI, uxI, uyI, uzI;
                    macro_at(idxI, rI, uxI, uyI, uzI);
                    const double pI = cs2 * (rI - 1.0);

                    const double Wm = 0.0;
                    const double Wp = uzI + pI / cs;

                    const double uzB = 0.5 * (Wp + Wm);
                    const double pB  = 0.5 * cs * (Wp - Wm);
                    const double rhoB  = 1.0 + pB / cs2;

                    double feqI[Q], feqB[Q];
                    feq_array(rI, uxI, uyI, uzI, feqI);
                    feq_array(rhoB, uxI, uyI, uzB, feqB);

                    const size_t baseI = idxI * Q;
                    for (int q = 0; q < Q; ++q)
                        rU[cnt++] = feqB[q] + (f_curr[baseI + q] - feqI[q]);
                }
        }

        MPI_Request req[4];

        int cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j)
                for (int q = 0; q < Q; ++q) sL[cnt++] = f_curr[loc_index(1, j, k, nx_t, ny_t) * Q + q];

        cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j)
                for (int q = 0; q < Q; ++q) sR[cnt++] = f_curr[loc_index(nx, j, k, nx_t, ny_t) * Q + q];

        MPI_Irecv(rL.data(), cx, MPI_DOUBLE, nbr[0], 10, cart, &req[0]);
        MPI_Irecv(rR.data(), cx, MPI_DOUBLE, nbr[1], 11, cart, &req[1]);
        MPI_Isend(sR.data(), cx, MPI_DOUBLE, nbr[1], 10, cart, &req[2]);
        MPI_Isend(sL.data(), cx, MPI_DOUBLE, nbr[0], 11, cart, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

        cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j)
                for (int q = 0; q < Q; ++q) f_curr[loc_index(0, j, k, nx_t, ny_t) * Q + q] = rL[cnt++];

        cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j)
                for (int q = 0; q < Q; ++q) f_curr[loc_index(nx + 1, j, k, nx_t, ny_t) * Q + q] = rR[cnt++];

        cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i)
                for (int q = 0; q < Q; ++q) sF[cnt++] = f_curr[loc_index(i, 1, k, nx_t, ny_t) * Q + q];

        cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i)
                for (int q = 0; q < Q; ++q) sB[cnt++] = f_curr[loc_index(i, ny, k, nx_t, ny_t) * Q + q];

        MPI_Irecv(rF.data(), cy, MPI_DOUBLE, nbr[2], 20, cart, &req[0]);
        MPI_Irecv(rB.data(), cy, MPI_DOUBLE, nbr[3], 21, cart, &req[1]);
        MPI_Isend(sB.data(), cy, MPI_DOUBLE, nbr[3], 20, cart, &req[2]);
        MPI_Isend(sF.data(), cy, MPI_DOUBLE, nbr[2], 21, cart, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

        cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i)
                for (int q = 0; q < Q; ++q) f_curr[loc_index(i, 0, k, nx_t, ny_t) * Q + q] = rF[cnt++];

        cnt = 0;
        for (int k = 1; k <= nz; ++k) for (int i = 0; i <= nx + 1; ++i)
                for (int q = 0; q < Q; ++q) f_curr[loc_index(i, ny + 1, k, nx_t, ny_t) * Q + q] = rB[cnt++];

        cnt = 0;
        for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i)
                for (int q = 0; q < Q; ++q) sD[cnt++] = f_curr[loc_index(i, j, 1, nx_t, ny_t) * Q + q];

        cnt = 0;
        for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i)
                for (int q = 0; q < Q; ++q) sU[cnt++] = f_curr[loc_index(i, j, nz, nx_t, ny_t) * Q + q];

        MPI_Irecv(rD.data(), cz, MPI_DOUBLE, nbr[4], 30, cart, &req[0]);
        MPI_Irecv(rU.data(), cz, MPI_DOUBLE, nbr[5], 31, cart, &req[1]);
        MPI_Isend(sU.data(), cz, MPI_DOUBLE, nbr[5], 30, cart, &req[2]);
        MPI_Isend(sD.data(), cz, MPI_DOUBLE, nbr[4], 31, cart, &req[3]);
        MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

        cnt = 0;
        for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i)
                for (int q = 0; q < Q; ++q) f_curr[loc_index(i, j, 0, nx_t, ny_t) * Q + q] = rD[cnt++];

        cnt = 0;
        for (int j = 0; j <= ny + 1; ++j) for (int i = 0; i <= nx + 1; ++i)
                for (int q = 0; q < Q; ++q) f_curr[loc_index(i, j, nz + 1, nx_t, ny_t) * Q + q] = rU[cnt++];
    }

    void step_collision_bgk()
    {
        const double omega = 1.0 / tau;

        #pragma omp parallel for collapse(3)
        for (int k = 1; k <= nz; ++k)
            for (int j = 1; j <= ny; ++j)
                for (int i = 1; i <= nx; ++i)
                {
                    size_t dst = loc_index(i, j, k, nx_t, ny_t);

                    double f_in[Q];
                    for (int q = 0; q < Q; ++q)
                    {
                        int ip = i - ex[q], jp = j - ey[q], kp = k - ez[q];
                        size_t src = loc_index(ip, jp, kp, nx_t, ny_t);

                        if (mask[src] == SOLID)
                        {
                            f_in[q] = f_curr[dst * Q + inv_q[q]];
                        }
                        else
                        {
                            f_in[q] = f_curr[src * Q + q];
                        }
                    }

                    if (mask[dst] == SOLID)
                    {
                        for (int q = 0; q < Q; ++q) f_next[dst * Q + q] = f_in[q];
                        rho[dst] = 1.0;
                        continue;
                    }

                    double r = 0, jx = 0, jy = 0, jz = 0;
                    for (int q = 0; q < Q; ++q)
                    {
                        r  += f_in[q];
                        jx += f_in[q] * ex[q];
                        jy += f_in[q] * ey[q];
                        jz += f_in[q] * ez[q];
                    }
                    if (r <= 0.0) r = 1e-12;
                    double ux = jx / r;
                    double uy = jy / r;
                    double uz = jz / r;

                    rho[dst] = r;

                    double sponge = sponge_sigma[dst];
                    for (int q = 0; q < Q; ++q)
                    {
                        double feq_q = feq(q, r, ux, uy, uz);

                        double f_post = f_in[q] - omega * (f_in[q] - feq_q);

                        if (sponge > 1e-12)
                        {
                            double feq0 = feq(q, 1.0, 0.0, 0.0, 0.0);
                            f_post = (1.0 - sponge) * f_post + sponge * feq0;
                        }

                        f_next[dst * Q + q] = f_post;
                    }
                }

        f_curr.swap(f_next);
    }

public:
    void write_vtk(int frame_id)
    {
        if (rank == 0) std::cout << "Write VTK frame " << frame_id << " ...\n";

        std::vector<float> local_p((size_t)nx * (size_t)ny * (size_t)nz);
        size_t ptr = 0;
        for (int k = 1; k <= nz; ++k) for (int j = 1; j <= ny; ++j) for (int i = 1; i <= nx; ++i)
                {
                    size_t idx = loc_index(i, j, k, nx_t, ny_t);
                    local_p[ptr++] = (float)(rho[idx] - 1.0);
                }

        if (rank == 0)
        {
            std::vector<float> global_p((size_t)NXg * (size_t)NYg * (size_t)NZg, 0.0f);

            ptr = 0;
            for (int k = 0; k < nz; ++k) for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i)
                    {
                        size_t gi = (size_t)(off_z + k) * (size_t)NXg * (size_t)NYg
                                    + (size_t)(off_y + j) * (size_t)NXg
                                    + (size_t)(off_x + i);
                        global_p[gi] = local_p[ptr++];
                    }

            for (int p = 1; p < size; ++p)
            {
                int c[3];
                MPI_Cart_coords(cart, p, 3, c);

                auto bl = [](int id, int pn, int n)
                {
                    return id * n / pn;
                };
                auto bs = [&](int id, int pn, int n)
                {
                    return bl(id + 1, pn, n) - bl(id, pn, n);
                };

                int ox = bl(c[0], dims[0], NXg), lx = bs(c[0], dims[0], NXg);
                int oy = bl(c[1], dims[1], NYg), ly = bs(c[1], dims[1], NYg);
                int oz = bl(c[2], dims[2], NZg), lz = bs(c[2], dims[2], NZg);

                std::vector<float> buf((size_t)lx * (size_t)ly * (size_t)lz);
                MPI_Recv(buf.data(), (int)buf.size(), MPI_FLOAT, p, 999, cart, MPI_STATUS_IGNORE);

                size_t bptr = 0;
                for (int kk = 0; kk < lz; ++kk) for (int jj = 0; jj < ly; ++jj) for (int ii = 0; ii < lx; ++ii)
                        {
                            size_t gi = (size_t)(oz + kk) * (size_t)NXg * (size_t)NYg
                                        + (size_t)(oy + jj) * (size_t)NXg
                                        + (size_t)(ox + ii);
                            global_p[gi] = buf[bptr++];
                        }
            }

            std::filesystem::create_directories("output_mpi");
            std::ostringstream fn;
            fn << "output_mpi/flight_" << std::setw(5) << std::setfill('0') << frame_id << ".vtk";
            std::ofstream f(fn.str());

            double ox = X_MIN, oy = Y_MIN, oz = Z_MIN;
            if (output_world_coords)
            {
                auto off = world_offset();
                ox += off[0];
                oy += off[1];
                oz += off[2];
            }

            f << "# vtk DataFile Version 3.0\nLBM Urban Acoustics V21\nASCII\n";
            f << "DATASET STRUCTURED_POINTS\n";
            f << "DIMENSIONS " << NXg << " " << NYg << " " << NZg << "\n";
            f << "ORIGIN " << ox << " " << oy << " " << oz << "\n";
            f << "SPACING " << units.dx_phys << " " << units.dx_phys << " " << units.dx_phys << "\n";
            f << "POINT_DATA " << (size_t)NXg*(size_t)NYg*(size_t)NZg << "\n";
            f << "SCALARS PressureAmp float 1\nLOOKUP_TABLE default\n";
            f << std::scientific << std::setprecision(6);
            for (auto v : global_p) f << v << "\n";
            f.close();
        }
        else
        {
            MPI_Send(local_p.data(), (int)local_p.size(), MPI_FLOAT, 0, 999, cart);
        }

        MPI_Barrier(cart);
    }

    void run(double T_total, int Frames)
    {
        double dt = units.get_dt_phys();
        int total_steps = (int)std::round(T_total / dt);
        int out_interval = std::max(1, total_steps / Frames);

        if (rank == 0)
        {
            std::cout << "Run: T_total=" << T_total << " s, Frames=" << Frames
                      << ", total_steps=" << total_steps << ", out_interval=" << out_interval << "\n";
        }

        int frame_id = 0;
        for (int s = 0; s <= total_steps; ++s)
        {
            double t = s * dt;

            apply_source(t);
            exchange_halos();
            step_collision_bgk();

            if (s % out_interval == 0)
            {
                write_vtk(frame_id++);
            }

            if (rank == 0 && (s % 200 == 0))
            {
                double pct = 100.0 * (double)s / (double)total_steps;
                std::cout << "Step " << s << "/" << total_steps << " (" << std::fixed << std::setprecision(1) << pct << "%)\r" << std::flush;
            }
        }
        if (rank == 0) std::cout << "\nDone.\n";
    }
};

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    LBMSolver s;
    s.DX = 1.5;

    s.stl_file = "csx_model_binary.stl";
    s.stl_sx   = -22248.0;
    s.stl_sy   = -4123.0;
    s.stl_sz   = -14.0;

    s.drone.sx = 912.0;
    s.drone.sy = 66.6;
    s.drone.sz = 86.0;
    s.drone.ex = 912.0;
    s.drone.ey = 1127.3;
    s.drone.ez = 86.0;
    s.drone.speed = 30.0;
    s.drone.amp   = 0.5;
    s.drone.freq  = 20.0;

    s.output_world_coords = true;

    s.init(argc, argv);

    s.run(3.0, 3);

    MPI_Finalize();
    return 0;
}