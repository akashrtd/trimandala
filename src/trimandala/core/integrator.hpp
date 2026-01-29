#pragma once
#include <array>
#include <cmath>

#include <vector>

namespace trimandala {

// Constants
constexpr double G = 1.0;
// Yoshida constants
const double w0 = -std::pow(2.0, 1.0 / 3.0) / (2.0 - std::pow(2.0, 1.0 / 3.0));
const double w1 = 1.0 / (2.0 - std::pow(2.0, 1.0 / 3.0));
const double c1 = w1 / 2.0;
const double c2 = (w0 + w1) / 2.0;
const double c3 = c2;
const double c4 = c1;
const double d1 = w1;
const double d2 = w0;
const double d3 = w1;

struct Body {
  double x, y, z;
  double vx, vy, vz;
  double mass;
};

class SymplecticIntegrator {
public:
  SymplecticIntegrator(const std::vector<double> &masses,
                       const std::vector<std::array<double, 3>> &pos,
                       const std::vector<std::array<double, 3>> &vel) {
    size_t n = masses.size();
    bodies.resize(n);
    acc.resize(n);

    for (size_t i = 0; i < n; ++i) {
      bodies[i].x = pos[i][0];
      bodies[i].y = pos[i][1];
      bodies[i].z = pos[i][2];
      bodies[i].vx = vel[i][0];
      bodies[i].vy = vel[i][1];
      bodies[i].vz = vel[i][2];
      bodies[i].mass = masses[i];
    }
  }

  void step(double dt) {
    // Yoshida 4th Order
    update_pos(c1, dt);
    update_vel(d1, dt);
    update_pos(c2, dt);
    update_vel(d2, dt);
    update_pos(c3, dt);
    update_vel(d3, dt);
    update_pos(c4, dt);
  }

  // Zero-Cost Readout (Pass pointer to buffer)
  void get_state(double *pos_ptr, double *vel_ptr) const {
    for (size_t i = 0; i < bodies.size(); ++i) {
      pos_ptr[i * 3 + 0] = bodies[i].x;
      pos_ptr[i * 3 + 1] = bodies[i].y;
      pos_ptr[i * 3 + 2] = bodies[i].z;

      vel_ptr[i * 3 + 0] = bodies[i].vx;
      vel_ptr[i * 3 + 1] = bodies[i].vy;
      vel_ptr[i * 3 + 2] = bodies[i].vz;
    }
  }

private:
  std::vector<Body> bodies;
  std::vector<std::array<double, 3>> acc;

  void compute_accelerations() {
    size_t n = bodies.size();
    // Reset acceleration
    for (size_t i = 0; i < n; ++i)
      acc[i] = {0, 0, 0};

    for (size_t i = 0; i < n; ++i) {
      for (size_t j = i + 1; j < n; ++j) {
        double dx = bodies[j].x - bodies[i].x;
        double dy = bodies[j].y - bodies[i].y;
        double dz = bodies[j].z - bodies[i].z;

        double dist_sq = dx * dx + dy * dy + dz * dz + 1e-10; // Softening
        double dist = std::sqrt(dist_sq);
        double f = G / (dist_sq * dist);

        double fx = f * dx;
        double fy = f * dy;
        double fz = f * dz;

        acc[i][0] += fx * bodies[j].mass;
        acc[i][1] += fy * bodies[j].mass;
        acc[i][2] += fz * bodies[j].mass;

        acc[j][0] -= fx * bodies[i].mass;
        acc[j][1] -= fy * bodies[i].mass;
        acc[j][2] -= fz * bodies[i].mass;
      }
    }
  }

  void update_pos(double c, double dt) {
    for (auto &b : bodies) {
      b.x += c * b.vx * dt;
      b.y += c * b.vy * dt;
      b.z += c * b.vz * dt;
    }
  }

  void update_vel(double d, double dt) {
    compute_accelerations();
    for (size_t i = 0; i < bodies.size(); ++i) {
      bodies[i].vx += d * acc[i][0] * dt;
      bodies[i].vy += d * acc[i][1] * dt;
      bodies[i].vz += d * acc[i][2] * dt;
    }
  }
};

} // namespace trimandala
