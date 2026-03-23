#pragma once

class void_p {
private:
  void *internal_ptr;

public:
  // 1. Default constructor
  void_p() : internal_ptr(nullptr) {}

  // 2. Implicit template constructor (Allows initialization/assignment from ANY T*) This allows: void_p p = new int;
  template <typename T>
  void_p(T *ptr) : internal_ptr(static_cast<void *>(ptr)) {}

  // 3. Implicit template conversion operator (Allows implicit conversion back to ANY T*) This allows: int* ip = p;
  // (where p holds an int*)
  template <typename T> operator T *() const {
    return static_cast<T *>(internal_ptr);
  }
};
