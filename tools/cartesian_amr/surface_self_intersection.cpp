// Detect non-adjacent triangle intersections in a binary PLY surface.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

using Vec = std::array<double, 3>;
using Tri = std::array<int32_t, 3>;

Vec sub(const Vec &a, const Vec &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Vec cross(const Vec &a, const Vec &b) {
  return {a[1] * b[2] - a[2] * b[1],
          a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
double dot(const Vec &a, const Vec &b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

struct Box { Vec lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300}; };
struct Node { Box box; int first = 0, last = 0, left = -1, right = -1; };

bool overlap(const Box &a, const Box &b, double tolerance) {
  for (int axis = 0; axis < 3; ++axis)
    if (a.hi[axis] < b.lo[axis] - tolerance ||
        b.hi[axis] < a.lo[axis] - tolerance) return false;
  return true;
}
bool adjacent(const Tri &a, const Tri &b) {
  for (int i : a) for (int j : b) if (i == j) return true;
  return false;
}
bool separated(const std::array<Vec, 3> &a, const std::array<Vec, 3> &b,
               const Vec &axis, double tolerance) {
  const double norm2 = dot(axis, axis);
  if (norm2 < 1e-28) return false;
  double amin = dot(a[0], axis), amax = amin;
  double bmin = dot(b[0], axis), bmax = bmin;
  for (int i = 1; i < 3; ++i) {
    const double ap = dot(a[i], axis), bp = dot(b[i], axis);
    amin = std::min(amin, ap); amax = std::max(amax, ap);
    bmin = std::min(bmin, bp); bmax = std::max(bmax, bp);
  }
  const double eps = tolerance * std::sqrt(norm2);
  return amax < bmin - eps || bmax < amin - eps;
}
bool intersects(const std::array<Vec, 3> &a, const std::array<Vec, 3> &b,
                double tolerance) {
  Vec ae[3], be[3];
  for (int i = 0; i < 3; ++i) {
    ae[i] = sub(a[(i + 1) % 3], a[i]);
    be[i] = sub(b[(i + 1) % 3], b[i]);
  }
  const Vec an = cross(ae[0], ae[1]), bn = cross(be[0], be[1]);
  if (separated(a, b, an, tolerance) || separated(a, b, bn, tolerance))
    return false;
  for (const Vec &ea : ae)
    for (const Vec &eb : be)
      if (separated(a, b, cross(ea, eb), tolerance)) return false;
  const Vec normal_cross = cross(an, bn);
  if (dot(normal_cross, normal_cross) <=
      1e-20 * std::max(dot(an, an) * dot(bn, bn), 1e-300)) {
    for (const Vec &edge : ae)
      if (separated(a, b, cross(an, edge), tolerance)) return false;
    for (const Vec &edge : be)
      if (separated(a, b, cross(an, edge), tolerance)) return false;
  }
  return true;
}

class Detector {
 public:
  std::vector<Vec> points, centers;
  std::vector<Tri> triangles;
  std::vector<Box> triangle_boxes;
  std::vector<int> order;
  std::vector<Node> nodes;
  std::vector<std::array<int32_t, 2>> collisions;
  double tolerance = 1e-12;
  size_t candidates = 0;

  int build(int first, int last) {
    Node node; node.first = first; node.last = last;
    for (int i = first; i < last; ++i) {
      const Box &box = triangle_boxes[order[i]];
      for (int axis = 0; axis < 3; ++axis) {
        node.box.lo[axis] = std::min(node.box.lo[axis], box.lo[axis]);
        node.box.hi[axis] = std::max(node.box.hi[axis], box.hi[axis]);
      }
    }
    const int id = static_cast<int>(nodes.size()); nodes.push_back(node);
    if (last - first <= 8) return id;
    int axis = 0;
    for (int j = 1; j < 3; ++j)
      if (node.box.hi[j] - node.box.lo[j] >
          node.box.hi[axis] - node.box.lo[axis]) axis = j;
    const int middle = (first + last) / 2;
    std::nth_element(order.begin() + first, order.begin() + middle,
                     order.begin() + last, [&](int a, int b) {
                       return centers[a][axis] < centers[b][axis];
                     });
    const int left = build(first, middle), right = build(middle, last);
    nodes[id].left = left; nodes[id].right = right; return id;
  }
  std::array<Vec, 3> xyz(int triangle) const {
    const Tri &t = triangles[triangle];
    return {points[t[0]], points[t[1]], points[t[2]]};
  }
  void test(int a, int b) {
    if (a == b || adjacent(triangles[a], triangles[b])) return;
    ++candidates;
    if (intersects(xyz(a), xyz(b), tolerance))
      collisions.push_back({static_cast<int32_t>(a), static_cast<int32_t>(b)});
  }
  void traverse(int a, int b) {
    const Node &na = nodes[a], &nb = nodes[b];
    if (!overlap(na.box, nb.box, tolerance)) return;
    if (a == b) {
      if (na.left < 0) {
        for (int i = na.first; i < na.last; ++i)
          for (int j = i + 1; j < na.last; ++j) test(order[i], order[j]);
      } else {
        traverse(na.left, na.left); traverse(na.left, na.right);
        traverse(na.right, na.right);
      }
    } else if (na.left < 0 && nb.left < 0) {
      for (int i = na.first; i < na.last; ++i)
        for (int j = nb.first; j < nb.last; ++j) test(order[i], order[j]);
    } else if (nb.left < 0 || (na.left >= 0 && na.last - na.first >= nb.last - nb.first)) {
      traverse(na.left, b); traverse(na.right, b);
    } else {
      traverse(a, nb.left); traverse(a, nb.right);
    }
  }
};

void read_ply(const std::string &path, Detector &detector) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open input PLY");
  std::string line; size_t nv = 0, nt = 0; bool binary = false;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line == "format binary_little_endian 1.0") binary = true;
    if (line.rfind("element vertex ", 0) == 0) nv = std::stoull(line.substr(15));
    if (line.rfind("element face ", 0) == 0) nt = std::stoull(line.substr(13));
    if (line == "end_header") break;
  }
  if (!binary || !nv || !nt) throw std::runtime_error("unsupported PLY header");
  detector.points.resize(nv);
  for (Vec &point : detector.points) {
    float value[3]; input.read(reinterpret_cast<char *>(value), sizeof(value));
    point = {value[0], value[1], value[2]};
  }
  detector.triangles.resize(nt);
  for (Tri &triangle : detector.triangles) {
    uint8_t count = 0; input.read(reinterpret_cast<char *>(&count), 1);
    if (count != 3) throw std::runtime_error("PLY contains a non-triangle face");
    input.read(reinterpret_cast<char *>(triangle.data()), 3 * sizeof(int32_t));
  }
  if (!input) throw std::runtime_error("truncated PLY file");
}

int main(int argc, char **argv) try {
  if (argc != 3) throw std::runtime_error("usage: check_surface INPUT.ply COLLISIONS.bin");
  const auto start = std::chrono::steady_clock::now();
  Detector detector; read_ply(argv[1], detector);
  const size_t count = detector.triangles.size();
  detector.triangle_boxes.resize(count); detector.centers.resize(count);
  detector.order.resize(count); std::iota(detector.order.begin(), detector.order.end(), 0);
  Box global;
  for (size_t i = 0; i < count; ++i) {
    Box &box = detector.triangle_boxes[i]; Vec center{};
    for (int vertex : detector.triangles[i]) for (int axis = 0; axis < 3; ++axis) {
      const double value = detector.points[vertex][axis];
      box.lo[axis] = std::min(box.lo[axis], value); box.hi[axis] = std::max(box.hi[axis], value);
      global.lo[axis] = std::min(global.lo[axis], value); global.hi[axis] = std::max(global.hi[axis], value);
      center[axis] += value / 3.0;
    }
    detector.centers[i] = center;
  }
  double diagonal2 = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const double d = global.hi[axis] - global.lo[axis]; diagonal2 += d * d;
  }
  detector.tolerance = std::max(1e-13 * std::sqrt(diagonal2), 1e-15);
  detector.nodes.reserve(2 * count); detector.build(0, static_cast<int>(count));
  detector.traverse(0, 0);
  std::sort(detector.collisions.begin(), detector.collisions.end());
  detector.collisions.erase(std::unique(detector.collisions.begin(), detector.collisions.end()), detector.collisions.end());
  std::ofstream output(argv[2], std::ios::binary);
  const uint64_t nc = detector.collisions.size();
  output.write(reinterpret_cast<const char *>(&nc), sizeof(nc));
  output.write(reinterpret_cast<const char *>(detector.collisions.data()), nc * sizeof(detector.collisions[0]));
  if (!output) throw std::runtime_error("cannot write collision pairs");
  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  std::cout << "{\"triangles\":" << count << ",\"candidate_pairs\":" << detector.candidates
            << ",\"collisions\":" << nc << ",\"seconds\":" << seconds << "}\n";
  return nc ? 1 : 0;
} catch (const std::exception &error) {
  std::cerr << "ERROR: " << error.what() << '\n'; return 2;
}
