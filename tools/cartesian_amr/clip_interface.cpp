// Clip an oriented triangle surface into Cartesian cells and retain one
// conservative polygon for each prism-front/Cartesian-cell intersection.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using Point = std::array<double, 3>;

Point operator+(Point a, const Point& b) {
    for (int axis = 0; axis < 3; ++axis) a[axis] += b[axis];
    return a;
}
Point operator-(Point a, const Point& b) {
    for (int axis = 0; axis < 3; ++axis) a[axis] -= b[axis];
    return a;
}
Point operator*(Point a, double scale) {
    for (double& value : a) value *= scale;
    return a;
}
Point cross(const Point& a, const Point& b) {
    return {a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}
double dot(const Point& a, const Point& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
double norm(const Point& a) { return std::sqrt(dot(a, a)); }

template <class T>
std::vector<T> read_binary(const std::filesystem::path& path, std::size_t count) {
    std::vector<T> values(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(count * sizeof(T)));
    if (!stream) throw std::runtime_error("cannot read " + path.string());
    return values;
}

template <class T>
void write_binary(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream stream(path, std::ios::binary);
    stream.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!stream) throw std::runtime_error("cannot write " + path.string());
}

std::vector<Point> clip_plane(const std::vector<Point>& input, int axis,
                              double boundary, bool keep_greater) {
    std::vector<Point> output;
    if (input.empty()) return output;
    output.reserve(input.size() + 1);
    auto signed_distance = [&](const Point& point) {
        const double value = point[axis] - boundary;
        return keep_greater ? value : -value;
    };
    Point previous = input.back();
    double previous_distance = signed_distance(previous);
    bool previous_inside = previous_distance >= 0.0;
    for (const Point& current : input) {
        const double current_distance = signed_distance(current);
        const bool current_inside = current_distance >= 0.0;
        if (current_inside != previous_inside) {
            const double denominator = previous_distance - current_distance;
            const double fraction = previous_distance / denominator;
            output.push_back(previous + (current - previous) * fraction);
        }
        if (current_inside) output.push_back(current);
        previous = current;
        previous_distance = current_distance;
        previous_inside = current_inside;
    }
    return output;
}

std::vector<Point> clip_box(std::vector<Point> polygon, const Point& low,
                            const Point& high) {
    for (int axis = 0; axis < 3 && !polygon.empty(); ++axis) {
        polygon = clip_plane(polygon, axis, low[axis], true);
        polygon = clip_plane(polygon, axis, high[axis], false);
    }
    return polygon;
}

struct Fragment {
    std::int32_t triangle;
    std::int32_t cartesian_cell;
    std::uint8_t count;
    Point centroid;
    Point area_vector;
};

std::uint64_t block_key(int i, int j, int k) {
    return static_cast<std::uint64_t>(i) |
           (static_cast<std::uint64_t>(j) << 20) |
           (static_cast<std::uint64_t>(k) << 40);
}

int main(int argc, char** argv) try {
    if (argc != 15) {
        throw std::runtime_error(
            "usage: clip_interface POINTS TRIANGLES BLOCK_COORDINATES "
            "BLOCK_LEVELS OUTPUT_DIR NV NT NB ROOT_X ROOT_Y ROOT_Z ROOT_SIZE "
            "MAXIMUM_LEVEL BLOCK_CELLS");
    }
    const std::filesystem::path point_path = argv[1];
    const std::filesystem::path triangle_path = argv[2];
    const std::filesystem::path coordinate_path = argv[3];
    const std::filesystem::path level_path = argv[4];
    const std::filesystem::path output = argv[5];
    const std::size_t vertex_count = std::stoull(argv[6]);
    const std::size_t triangle_count = std::stoull(argv[7]);
    const std::size_t block_count = std::stoull(argv[8]);
    const Point root_low = {std::stod(argv[9]), std::stod(argv[10]),
                            std::stod(argv[11])};
    const double root_size = std::stod(argv[12]);
    const int maximum_level = std::stoi(argv[13]);
    const int block_cells = std::stoi(argv[14]);
    if (maximum_level < 1 || maximum_level > 19 || block_cells < 1) {
        throw std::runtime_error("invalid Cartesian dimensions");
    }
    const auto points = read_binary<Point>(point_path, vertex_count);
    const auto triangles = read_binary<std::array<std::int32_t, 3>>(
        triangle_path, triangle_count);
    const auto block_coordinates = read_binary<std::array<std::int32_t, 3>>(
        coordinate_path, block_count);
    const auto block_levels = read_binary<std::int16_t>(level_path, block_count);
    std::vector<std::unordered_map<std::uint64_t, std::int32_t>> blocks_by_level(
        static_cast<std::size_t>(maximum_level + 1));
    for (std::size_t block = 0; block < block_count; ++block) {
        const int level = block_levels[block];
        if (level < 0 || level > maximum_level) {
            throw std::runtime_error("invalid leaf-block level");
        }
        const auto coordinate = block_coordinates[block];
        blocks_by_level[static_cast<std::size_t>(level)].emplace(
            block_key(coordinate[0], coordinate[1], coordinate[2]),
            static_cast<std::int32_t>(block));
    }
    const int block_dimension = 1 << maximum_level;
    const double finest_block_size = root_size / block_dimension;
    const auto find_leaf_block = [&](int i, int j, int k) {
        for (int level = maximum_level; level >= 0; --level) {
            const int span = 1 << (maximum_level - level);
            const int oi = (i / span) * span;
            const int oj = (j / span) * span;
            const int ok = (k / span) * span;
            const auto& lookup = blocks_by_level[static_cast<std::size_t>(level)];
            const auto found = lookup.find(block_key(oi, oj, ok));
            if (found != lookup.end()) return found->second;
        }
        return static_cast<std::int32_t>(-1);
    };
    std::vector<Fragment> fragments;
    std::vector<Point> fragment_points;
    fragments.reserve(triangle_count * 3);
    fragment_points.reserve(triangle_count * 12);
    std::vector<double> covered_area(triangle_count, 0.0);
    double original_area_total = 0.0;
    std::size_t candidate_boxes = 0;
    std::size_t missing_leaf_blocks = 0;

    for (std::size_t triangle = 0; triangle < triangle_count; ++triangle) {
        const auto ids = triangles[triangle];
        const Point a = points[ids[0]], b = points[ids[1]], c = points[ids[2]];
        const Point original_area_vector = cross(b - a, c - a) * 0.5;
        const double original_area = norm(original_area_vector);
        if (!(original_area > 0.0)) throw std::runtime_error("zero-area triangle");
        original_area_total += original_area;
        const Point unit_normal = original_area_vector * (1.0 / original_area);
        Point minimum, maximum;
        std::array<int, 3> first_block, last_block;
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] = std::min({a[axis], b[axis], c[axis]});
            maximum[axis] = std::max({a[axis], b[axis], c[axis]});
            first_block[axis] = std::clamp(
                static_cast<int>(std::floor((minimum[axis] - root_low[axis]) /
                                            finest_block_size)), 0,
                block_dimension - 1);
            last_block[axis] = std::clamp(
                static_cast<int>(std::floor((maximum[axis] - root_low[axis]) /
                                            finest_block_size)), 0,
                block_dimension - 1);
        }
        std::vector<std::int32_t> candidate_blocks;
        for (int k = first_block[2]; k <= last_block[2]; ++k) {
            for (int j = first_block[1]; j <= last_block[1]; ++j) {
                for (int i = first_block[0]; i <= last_block[0]; ++i) {
                    const std::int32_t block = find_leaf_block(i, j, k);
                    if (block < 0) {
                        ++missing_leaf_blocks;
                    } else if (std::find(candidate_blocks.begin(), candidate_blocks.end(),
                                         block) == candidate_blocks.end()) {
                        candidate_blocks.push_back(block);
                    }
                }
            }
        }
        for (const std::int32_t block : candidate_blocks) {
            const int span = 1 << (maximum_level - block_levels[block]);
            const auto coordinate = block_coordinates[block];
            const Point block_low = {
                root_low[0] + coordinate[0] * finest_block_size,
                root_low[1] + coordinate[1] * finest_block_size,
                root_low[2] + coordinate[2] * finest_block_size,
            };
            // Clip every surface polygon on the global finest-cell lattice,
            // even when its owning procedural cell belongs to a coarser leaf.
            // The extra fragment subdivision makes coarse/fine cut faces share
            // identical atomic subfaces while retaining one coarse control.
            const double cell_size = finest_block_size / block_cells;
            const int clipping_cells = block_cells * span;
            const double probe_distance = 1e-7 * cell_size;
            std::array<int, 3> first, last;
            for (int axis = 0; axis < 3; ++axis) {
                first[axis] = std::clamp(
                    static_cast<int>(std::floor(
                        (minimum[axis] - block_low[axis]) / cell_size)),
                    0, clipping_cells - 1);
                last[axis] = std::clamp(
                    static_cast<int>(std::floor(
                        (maximum[axis] - block_low[axis]) / cell_size)),
                    0, clipping_cells - 1);
            }
            for (int k = first[2]; k <= last[2]; ++k) {
                for (int j = first[1]; j <= last[1]; ++j) {
                    for (int i = first[0]; i <= last[0]; ++i) {
                    ++candidate_boxes;
                    Point low = {block_low[0] + i * cell_size,
                                 block_low[1] + j * cell_size,
                                 block_low[2] + k * cell_size};
                    Point high = {low[0] + cell_size, low[1] + cell_size,
                                  low[2] + cell_size};
                    auto polygon = clip_box({a, b, c}, low, high);
                    if (polygon.size() < 3) continue;
                    Point area_vector = {0.0, 0.0, 0.0};
                    Point centroid = {0.0, 0.0, 0.0};
                    double polygon_area = 0.0;
                    for (std::size_t q = 1; q + 1 < polygon.size(); ++q) {
                        const Point vector = cross(polygon[q] - polygon[0],
                                                   polygon[q + 1] - polygon[0]) * 0.5;
                        const double area = norm(vector);
                        area_vector = area_vector + vector;
                        centroid = centroid +
                            (polygon[0] + polygon[q] + polygon[q + 1]) * (area / 3.0);
                        polygon_area += area;
                    }
                    if (!(polygon_area > 1e-20 * cell_size * cell_size)) continue;
                    centroid = centroid * (1.0 / polygon_area);
                    if (dot(area_vector, original_area_vector) < 0.0) {
                        std::reverse(polygon.begin(), polygon.end());
                        area_vector = area_vector * -1.0;
                    }
                    const Point probe = centroid + unit_normal * probe_distance;
                    std::array<int, 3> probe_cell;
                    for (int axis = 0; axis < 3; ++axis) {
                        probe_cell[axis] = static_cast<int>(std::floor(
                            (probe[axis] - block_low[axis]) / cell_size));
                    }
                    const int local_i = i / span;
                    const int local_j = j / span;
                    const int local_k = k / span;
                    if (probe_cell[0] < 0 || probe_cell[1] < 0 ||
                        probe_cell[2] < 0 || probe_cell[0] >= clipping_cells ||
                        probe_cell[1] >= clipping_cells ||
                        probe_cell[2] >= clipping_cells ||
                        probe_cell[0] / span != local_i ||
                        probe_cell[1] / span != local_j ||
                        probe_cell[2] / span != local_k) continue;
                    const int local_cell =
                        (local_k * block_cells + local_j) * block_cells + local_i;
                    const std::int32_t cartesian_cell =
                        block * block_cells * block_cells * block_cells +
                        local_cell;
                    if (polygon.size() > 255) {
                        throw std::runtime_error("fragment vertex-count overflow");
                    }
                    fragments.push_back({static_cast<std::int32_t>(triangle),
                                         cartesian_cell,
                                         static_cast<std::uint8_t>(polygon.size()),
                                         centroid, area_vector});
                    fragment_points.insert(fragment_points.end(), polygon.begin(),
                                           polygon.end());
                    covered_area[triangle] += norm(area_vector);
                }
            }
        }
        }
    }
    if (missing_leaf_blocks) {
        throw std::runtime_error("surface fragment reached outside the leaf-block tree");
    }
    double maximum_relative_area_error = 0.0;
    double covered_area_total = 0.0;
    for (std::size_t triangle = 0; triangle < triangle_count; ++triangle) {
        const auto ids = triangles[triangle];
        const double area = 0.5 * norm(cross(points[ids[1]] - points[ids[0]],
                                             points[ids[2]] - points[ids[0]]));
        maximum_relative_area_error = std::max(
            maximum_relative_area_error, std::abs(covered_area[triangle] - area) / area);
        covered_area_total += covered_area[triangle];
    }
    std::filesystem::create_directories(output);
    std::vector<std::int32_t> fragment_triangles, fragment_cells;
    std::vector<std::uint8_t> fragment_counts;
    std::vector<Point> fragment_centroids, fragment_area_vectors;
    fragment_triangles.reserve(fragments.size());
    fragment_cells.reserve(fragments.size());
    fragment_counts.reserve(fragments.size());
    fragment_centroids.reserve(fragments.size());
    fragment_area_vectors.reserve(fragments.size());
    for (const Fragment& fragment : fragments) {
        fragment_triangles.push_back(fragment.triangle);
        fragment_cells.push_back(fragment.cartesian_cell);
        fragment_counts.push_back(fragment.count);
        fragment_centroids.push_back(fragment.centroid);
        fragment_area_vectors.push_back(fragment.area_vector);
    }
    write_binary(output / "fragment_triangle.i32", fragment_triangles);
    write_binary(output / "fragment_cartesian_cell.i32", fragment_cells);
    write_binary(output / "fragment_count.u8", fragment_counts);
    write_binary(output / "fragment_points.f64", fragment_points);
    write_binary(output / "fragment_centroid.f64", fragment_centroids);
    write_binary(output / "fragment_area_vector.f64", fragment_area_vectors);
    std::cout << std::setprecision(17)
              << "{\"status\":\"INTERFACE_CLIPPED\",\"candidate_triangle_boxes\":"
              << candidate_boxes << ",\"fragments\":" << fragments.size()
              << ",\"fragment_vertices\":" << fragment_points.size()
              << ",\"original_area\":" << original_area_total
              << ",\"covered_area\":" << covered_area_total
              << ",\"relative_total_area_error\":"
              << std::abs(covered_area_total - original_area_total) / original_area_total
              << ",\"maximum_triangle_relative_area_error\":"
              << maximum_relative_area_error << "}\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 2;
}
