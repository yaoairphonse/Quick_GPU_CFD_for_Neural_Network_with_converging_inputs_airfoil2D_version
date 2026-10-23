// array.h -- The only small utility class: allocate and release a double array.
// GPU data directives (acc enter data / acc exit data) can be added here centrally.
#ifndef CFD_ARRAY_H
#define CFD_ARRAY_H

#include <cstddef>
#include <cstdio>
#include <cstdlib>

namespace cfd {

class Array {
 public:
  Array() : p_(nullptr), n_(0) {}
  explicit Array(size_t n) : p_(nullptr), n_(0) { allocate(n); }
  ~Array() { release(); }

  Array(const Array&) = delete;
  Array& operator=(const Array&) = delete;
  Array(Array&& o) noexcept : p_(o.p_), n_(o.n_) {
    o.p_ = nullptr;
    o.n_ = 0;
  }
  Array& operator=(Array&& o) noexcept {
    if (this != &o) {
      release();
      p_ = o.p_;
      n_ = o.n_;
      o.p_ = nullptr;
      o.n_ = 0;
    }
    return *this;
  }

  // Allocate and zero the array. Abort on failure; exceptions are not used.
  void allocate(size_t n) {
    release();
    p_ = static_cast<double*>(std::calloc(n > 0 ? n : 1, sizeof(double)));
    if (p_ == nullptr) {
      std::fprintf(stderr, "Array::allocate: out of memory (%zu doubles)\n", n);
      std::abort();
    }
    n_ = n;
  }
  void release() {
    std::free(p_);
    p_ = nullptr;
    n_ = 0;
  }

  double* data() { return p_; }
  const double* data() const { return p_; }
  size_t size() const { return n_; }
  double& operator[](size_t i) { return p_[i]; }
  const double& operator[](size_t i) const { return p_[i]; }

 private:
  double* p_;
  size_t n_;
};

}  // namespace cfd

#endif  // CFD_ARRAY_H
