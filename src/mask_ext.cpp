#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

namespace py = pybind11;

constexpr int kVerbPick = 0;
constexpr int kVerbPlace = 1;
constexpr int kVerbFlip = 2;
constexpr int kVerbSteal = 3;

template <typename T>
inline T read_scalar(const void* base, at::ScalarType type, int64_t index) {
  switch (type) {
    case at::ScalarType::Byte:
      return static_cast<T>(reinterpret_cast<const uint8_t*>(base)[index]);
    case at::ScalarType::Char:
      return static_cast<T>(reinterpret_cast<const int8_t*>(base)[index]);
    case at::ScalarType::Short:
      return static_cast<T>(reinterpret_cast<const int16_t*>(base)[index]);
    case at::ScalarType::Int:
      return static_cast<T>(reinterpret_cast<const int32_t*>(base)[index]);
    case at::ScalarType::Long:
      return static_cast<T>(reinterpret_cast<const int64_t*>(base)[index]);
    case at::ScalarType::Float:
      return static_cast<T>(reinterpret_cast<const float*>(base)[index]);
    case at::ScalarType::Double:
      return static_cast<T>(reinterpret_cast<const double*>(base)[index]);
    case at::ScalarType::Bool:
      return static_cast<T>(reinterpret_cast<const bool*>(base)[index]);
    default:
      TORCH_CHECK(false, "Unsupported dtype for scalar read");
  }
}

struct Reader1D {
  const void* data{nullptr};
  at::ScalarType type{at::ScalarType::Undefined};
  int64_t size0{0};

  Reader1D() = default;
  explicit Reader1D(const torch::Tensor& t) { reset(t); }

  void reset(const torch::Tensor& t) {
    TORCH_CHECK(t.defined(), "Tensor is undefined");
    TORCH_CHECK(t.device().is_cpu(), "Tensor must be on CPU");
    TORCH_CHECK(t.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(t.dim() == 1, "Expected 1D tensor");
    data = t.data_ptr();
    type = t.scalar_type();
    size0 = t.size(0);
  }

  template <typename T>
  T get(int64_t i) const {
    TORCH_CHECK(i >= 0 && i < size0, "Index out of range in Reader1D");
    return read_scalar<T>(data, type, i);
  }
};

struct Reader2D {
  const void* data{nullptr};
  at::ScalarType type{at::ScalarType::Undefined};
  int64_t size0{0};
  int64_t size1{0};

  Reader2D() = default;
  explicit Reader2D(const torch::Tensor& t) { reset(t); }

  void reset(const torch::Tensor& t) {
    TORCH_CHECK(t.defined(), "Tensor is undefined");
    TORCH_CHECK(t.device().is_cpu(), "Tensor must be on CPU");
    TORCH_CHECK(t.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(t.dim() == 2, "Expected 2D tensor");
    data = t.data_ptr();
    type = t.scalar_type();
    size0 = t.size(0);
    size1 = t.size(1);
  }

  template <typename T>
  T get(int64_t i, int64_t j) const {
    TORCH_CHECK(
        i >= 0 && i < size0 && j >= 0 && j < size1,
        "Index out of range in Reader2D");
    return read_scalar<T>(data, type, i * size1 + j);
  }
};

struct Reader3D {
  const void* data{nullptr};
  at::ScalarType type{at::ScalarType::Undefined};
  int64_t size0{0};
  int64_t size1{0};
  int64_t size2{0};

  Reader3D() = default;
  explicit Reader3D(const torch::Tensor& t) { reset(t); }

  void reset(const torch::Tensor& t) {
    TORCH_CHECK(t.defined(), "Tensor is undefined");
    TORCH_CHECK(t.device().is_cpu(), "Tensor must be on CPU");
    TORCH_CHECK(t.is_contiguous(), "Tensor must be contiguous");
    TORCH_CHECK(t.dim() == 3, "Expected 3D tensor");
    data = t.data_ptr();
    type = t.scalar_type();
    size0 = t.size(0);
    size1 = t.size(1);
    size2 = t.size(2);
  }

  template <typename T>
  T get(int64_t i, int64_t j, int64_t k) const {
    TORCH_CHECK(
        i >= 0 && i < size0 && j >= 0 && j < size1 && k >= 0 && k < size2,
        "Index out of range in Reader3D");
    const int64_t offset = (i * size1 + j) * size2 + k;
    return read_scalar<T>(data, type, offset);
  }
};

inline int64_t ensure_mask_has_true(
    bool* mask_row,
    int64_t len,
    const float* logits_row,
    int64_t start_index) {
  for (int64_t i = start_index; i < len; ++i) {
    if (mask_row[i]) {
      return i;
    }
  }
  TORCH_CHECK(
      start_index < len,
      "ensure_mask_has_true received invalid start index");
  int64_t best = start_index;
  float best_val = logits_row[start_index];
  for (int64_t i = start_index + 1; i < len; ++i) {
    const float val = logits_row[i];
    if (val > best_val) {
      best_val = val;
      best = i;
    }
  }
  mask_row[best] = 1;
  return best;
}

inline void apply_big_neg(
    float* logits_row,
    const bool* mask_row,
    int64_t len,
    float big_neg) {
  for (int64_t i = 0; i < len; ++i) {
    if (!mask_row[i]) {
      logits_row[i] = big_neg;
    }
  }
}

inline int32_t clamp_positive(int32_t v) {
  return v > 0 ? v : 0;
}

inline int32_t clamp_max(int32_t v, int32_t hi) {
  return v < hi ? v : hi;
}

struct RowData {
  int32_t cap;
  int32_t total_inv;
  int32_t nest_cap_left;
  std::vector<int32_t> inv_colors;
  std::vector<int32_t> pickup_best;
  std::vector<int32_t> max_pick_by_color;
  std::vector<int32_t> pickup_stock;
  std::vector<int32_t> pantry_room;
  std::vector<int32_t> pantry_stock;
  std::vector<int32_t> pantry_yellow;
  std::vector<int32_t> max_steal_color;
  int32_t max_pantry_room;
  int32_t max_pick_qty;
  int32_t max_place_qty;
  int32_t max_flip_qty;
  int32_t max_steal_qty;
  bool pick_valid;
  bool place_valid;
  bool flip_valid;
  bool steal_valid;

  RowData(int64_t n_colors, int64_t n_pickups, int64_t n_pantries)
      : cap(0),
        total_inv(0),
        nest_cap_left(0),
        inv_colors(n_colors, 0),
        pickup_best(n_pickups, 0),
        max_pick_by_color(n_colors, 0),
        pickup_stock(n_pickups * n_colors, 0),
        pantry_room(n_pantries, 0),
        pantry_stock(n_pantries * n_colors, 0),
        pantry_yellow(n_pantries, 0),
        max_steal_color(n_colors, 0),
        max_pantry_room(0),
        max_pick_qty(0),
        max_place_qty(0),
        max_flip_qty(0),
        max_steal_qty(0),
        pick_valid(false),
        place_valid(false),
        flip_valid(false),
        steal_valid(false) {}
};

RowData compute_row_data(
    int64_t b,
    const Reader1D& cap_reader,
    const Reader1D& inv_sum_reader,
    const Reader1D& nest_reader,
    const Reader2D& inv_reader,
    const Reader3D& pick_reader,
    const Reader3D& pan_reader,
    const Reader2D& pan_room_reader,
    int64_t n_colors,
    int64_t n_pickups,
    int64_t n_pantries,
    int color_yellow,
    int64_t nest_cap,
    int64_t nest_blue_id,
    bool allow_steal,
    bool can_flip,
    int64_t n_verbs) {
  RowData row(n_colors, n_pickups, n_pantries);

  row.cap = clamp_positive(cap_reader.get<int32_t>(b));
  row.total_inv = clamp_positive(inv_sum_reader.get<int32_t>(b));
  int32_t nest_count = clamp_positive(nest_reader.get<int32_t>(b));
  if (nest_cap > 0 && nest_blue_id >= 0) {
    int64_t diff = nest_cap - static_cast<int64_t>(nest_count);
    row.nest_cap_left = diff > 0 ? static_cast<int32_t>(diff) : 0;
  }

  for (int64_t c = 0; c < n_colors; ++c) {
    row.inv_colors[c] = clamp_positive(inv_reader.get<int32_t>(b, c));
  }

  for (int64_t p = 0; p < n_pickups; ++p) {
    int32_t best = 0;
    for (int64_t c = 0; c < n_colors; ++c) {
      int32_t stock = clamp_positive(pick_reader.get<int32_t>(b, p, c));
      row.pickup_stock[p * n_colors + c] = stock;
      if (stock > best) {
        best = stock;
      }
      if (stock > row.max_pick_by_color[c]) {
        row.max_pick_by_color[c] = stock;
      }
      if (row.cap > 0) {
        int32_t limit = stock < row.cap ? stock : row.cap;
        if (limit > row.max_pick_qty) {
          row.max_pick_qty = limit;
        }
      }
    }
    row.pickup_best[p] = best;
  }

  for (int64_t p = 0; p < n_pantries; ++p) {
    int32_t room = clamp_positive(pan_room_reader.get<int32_t>(b, p));
    row.pantry_room[p] = room;
    if (room > row.max_pantry_room) {
      row.max_pantry_room = room;
    }
    for (int64_t c = 0; c < n_colors; ++c) {
      int32_t stock = clamp_positive(pan_reader.get<int32_t>(b, p, c));
      row.pantry_stock[p * n_colors + c] = stock;
      if (stock > row.max_steal_color[c]) {
        row.max_steal_color[c] = stock;
      }
      if (row.cap > 0) {
        int32_t limit = stock < row.cap ? stock : row.cap;
        if (limit > row.max_steal_qty) {
          row.max_steal_qty = limit;
        }
      }
    }
    if (color_yellow >= 0 && color_yellow < n_colors) {
      row.pantry_yellow[p] = row.pantry_stock[p * n_colors + color_yellow];
    }
  }

  for (int64_t p = 0; p < n_pantries; ++p) {
    int32_t room = row.pantry_room[p];
    if (room <= 0) {
      continue;
    }
    for (int64_t c = 0; c < n_colors; ++c) {
      int32_t inv_val = row.inv_colors[c];
      if (inv_val <= 0) {
        continue;
      }
      int32_t limit = room < inv_val ? room : inv_val;
      if (limit > row.max_place_qty) {
        row.max_place_qty = limit;
      }
    }
  }

  if (row.nest_cap_left > 0) {
    for (int64_t c = 0; c < n_colors; ++c) {
      int32_t inv_val = row.inv_colors[c];
      if (inv_val <= 0) {
        continue;
      }
      int32_t limit = row.nest_cap_left < inv_val ? row.nest_cap_left : inv_val;
      if (limit > row.max_place_qty) {
        row.max_place_qty = limit;
      }
    }
  }

  for (int64_t c = 0; c < n_colors; ++c) {
    int32_t other = row.total_inv - row.inv_colors[c];
    if (other > row.max_flip_qty) {
      row.max_flip_qty = other;
    }
  }

  bool any_pick_stock =
      std::any_of(row.pickup_best.begin(), row.pickup_best.end(), [](int32_t v) { return v > 0; });
  bool any_pantry_yellow = std::any_of(
      row.pantry_yellow.begin(), row.pantry_yellow.end(), [](int32_t v) { return v > 0; });

  row.pick_valid = (kVerbPick < n_verbs) && row.cap > 0 && any_pick_stock;
  row.place_valid =
      (kVerbPlace < n_verbs) && row.total_inv > 0 && (row.max_pantry_room > 0 || row.nest_cap_left > 0);
  row.flip_valid = (kVerbFlip < n_verbs) && can_flip && row.max_flip_qty > 0;
  row.steal_valid =
      (kVerbSteal < n_verbs) && allow_steal && row.cap > 0 && any_pantry_yellow;

  return row;
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
build_masks_and_apply_logits(
    const torch::Tensor& verb_logits,
    const torch::Tensor& node_logits,
    const torch::Tensor& color_logits,
    const torch::Tensor& qty_logits,
    const torch::Tensor& blue_node,
    const torch::Tensor& inv_b,
    const torch::Tensor& inv_sum,
    const torch::Tensor& cap_left,
    const torch::Tensor& pan,
    const torch::Tensor& pick,
    const torch::Tensor& pan_room,
    const torch::Tensor& pan_sum,
    const torch::Tensor& pick_sum,
    const torch::Tensor& nest_blue,
    const torch::Tensor& pantry_idx,
    const torch::Tensor& pickup_idx,
    const torch::Tensor& pantry_nodes,
    const torch::Tensor& pickup_nodes,
    int64_t n_verbs,
    int64_t n_nodes,
    int64_t n_colors,
    int64_t Q,
    int64_t capacity,
    int64_t pantry_cap,
    int64_t nest_cap,
    int64_t nest_blue_id,
    bool allow_steal,
    bool can_flip,
    double big_neg,
    bool apply_to_logits,
    bool flip_node_all,
    const c10::optional<torch::Tensor>& verb_sel_opt,
    const c10::optional<torch::Tensor>& node_sel_opt,
    const c10::optional<torch::Tensor>& color_sel_opt) {
  (void)pan_sum;
  (void)pick_sum;
  (void)capacity;
  (void)pantry_cap;
  (void)pantry_nodes;
  (void)pickup_nodes;

  TORCH_CHECK(verb_logits.device().is_cpu(), "verb_logits must be on CPU");
  TORCH_CHECK(node_logits.device().is_cpu(), "node_logits must be on CPU");
  TORCH_CHECK(color_logits.device().is_cpu(), "color_logits must be on CPU");
  TORCH_CHECK(qty_logits.device().is_cpu(), "qty_logits must be on CPU");
  TORCH_CHECK(verb_logits.scalar_type() == torch::kFloat32, "verb_logits must be float32");
  TORCH_CHECK(node_logits.scalar_type() == torch::kFloat32, "node_logits must be float32");
  TORCH_CHECK(color_logits.scalar_type() == torch::kFloat32, "color_logits must be float32");
  TORCH_CHECK(qty_logits.scalar_type() == torch::kFloat32, "qty_logits must be float32");
  TORCH_CHECK(verb_logits.is_contiguous(), "verb_logits must be contiguous");
  TORCH_CHECK(node_logits.is_contiguous(), "node_logits must be contiguous");
  TORCH_CHECK(color_logits.is_contiguous(), "color_logits must be contiguous");
  TORCH_CHECK(qty_logits.is_contiguous(), "qty_logits must be contiguous");

  TORCH_CHECK(verb_logits.dim() == 2, "verb_logits must be 2D");
  TORCH_CHECK(node_logits.dim() == 2, "node_logits must be 2D");
  TORCH_CHECK(color_logits.dim() == 2, "color_logits must be 2D");
  TORCH_CHECK(qty_logits.dim() == 2, "qty_logits must be 2D");

  const int64_t B = verb_logits.size(0);
  TORCH_CHECK(node_logits.size(0) == B, "node_logits batch mismatch");
  TORCH_CHECK(color_logits.size(0) == B, "color_logits batch mismatch");
  TORCH_CHECK(qty_logits.size(0) == B, "qty_logits batch mismatch");

  TORCH_CHECK(verb_logits.size(1) == n_verbs, "verb_logits size(1) mismatch");
  TORCH_CHECK(node_logits.size(1) == n_nodes, "node_logits size(1) mismatch");
  TORCH_CHECK(color_logits.size(1) == n_colors, "color_logits size(1) mismatch");
  TORCH_CHECK(qty_logits.size(1) == Q, "qty_logits size(1) mismatch");

  TORCH_CHECK(blue_node.dim() == 1 && blue_node.size(0) == B, "blue_node shape mismatch");
  TORCH_CHECK(inv_b.dim() == 2 && inv_b.size(0) == B && inv_b.size(1) == n_colors, "inv_b shape mismatch");
  TORCH_CHECK(inv_sum.dim() == 1 && inv_sum.size(0) == B, "inv_sum shape mismatch");
  TORCH_CHECK(cap_left.dim() == 1 && cap_left.size(0) == B, "cap_left shape mismatch");
  TORCH_CHECK(nest_blue.dim() == 1 && nest_blue.size(0) == B, "nest_blue shape mismatch");

  TORCH_CHECK(pan.dim() == 3 && pan.size(0) == B && pan.size(2) == n_colors, "pan shape mismatch");
  TORCH_CHECK(pick.dim() == 3 && pick.size(0) == B && pick.size(2) == n_colors, "pick shape mismatch");
  TORCH_CHECK(pan_room.dim() == 2 && pan_room.size(0) == B && pan_room.size(1) == pan.size(1), "pan_room shape mismatch");

  TORCH_CHECK(pantry_idx.dim() == 1 && pantry_idx.size(0) == n_nodes, "pantry_idx shape mismatch");
  TORCH_CHECK(pickup_idx.dim() == 1 && pickup_idx.size(0) == n_nodes, "pickup_idx shape mismatch");

  const int64_t n_pantries = pan.size(1);
  const int64_t n_pickups = pick.size(1);

  bool have_verb_sel = verb_sel_opt.has_value() && verb_sel_opt->defined();
  bool have_node_sel = node_sel_opt.has_value() && node_sel_opt->defined();
  bool have_color_sel = color_sel_opt.has_value() && color_sel_opt->defined();

  torch::Tensor verb_sel_tensor;
  const int64_t* verb_sel_ptr = nullptr;
  if (have_verb_sel) {
    verb_sel_tensor = verb_sel_opt.value();
    TORCH_CHECK(
        verb_sel_tensor.device().is_cpu(), "verb_sel must be on CPU");
    TORCH_CHECK(
        verb_sel_tensor.is_contiguous(), "verb_sel must be contiguous");
    TORCH_CHECK(
        verb_sel_tensor.dim() == 1 && verb_sel_tensor.size(0) == B,
        "verb_sel shape mismatch");
    TORCH_CHECK(
        verb_sel_tensor.scalar_type() == torch::kLong,
        "verb_sel must be int64");
    verb_sel_ptr = verb_sel_tensor.data_ptr<int64_t>();
  }

  torch::Tensor node_sel_tensor;
  const int64_t* node_sel_ptr = nullptr;
  if (have_node_sel) {
    node_sel_tensor = node_sel_opt.value();
    TORCH_CHECK(
        node_sel_tensor.device().is_cpu(), "node_sel must be on CPU");
    TORCH_CHECK(
        node_sel_tensor.is_contiguous(), "node_sel must be contiguous");
    TORCH_CHECK(
        node_sel_tensor.dim() == 1 && node_sel_tensor.size(0) == B,
        "node_sel shape mismatch");
    TORCH_CHECK(
        node_sel_tensor.scalar_type() == torch::kLong,
        "node_sel must be int64");
    node_sel_ptr = node_sel_tensor.data_ptr<int64_t>();
  }

  torch::Tensor color_sel_tensor;
  const int64_t* color_sel_ptr = nullptr;
  if (have_color_sel) {
    color_sel_tensor = color_sel_opt.value();
    TORCH_CHECK(
        color_sel_tensor.device().is_cpu(), "color_sel must be on CPU");
    TORCH_CHECK(
        color_sel_tensor.is_contiguous(), "color_sel must be contiguous");
    TORCH_CHECK(
        color_sel_tensor.dim() == 1 && color_sel_tensor.size(0) == B,
        "color_sel shape mismatch");
    TORCH_CHECK(
        color_sel_tensor.scalar_type() == torch::kLong,
        "color_sel must be int64");
    color_sel_ptr = color_sel_tensor.data_ptr<int64_t>();
  }

  Reader1D blue_reader(blue_node);
  Reader2D inv_reader(inv_b);
  Reader1D inv_sum_reader(inv_sum);
  Reader1D cap_reader(cap_left);
  Reader3D pan_reader(pan);
  Reader3D pick_reader(pick);
  Reader2D pan_room_reader(pan_room);
  Reader1D nest_reader(nest_blue);

  const auto pantry_idx_ptr = pantry_idx.data_ptr<int64_t>();
  const auto pickup_idx_ptr = pickup_idx.data_ptr<int64_t>();

  const float* verb_in = verb_logits.data_ptr<float>();
  const float* node_in = node_logits.data_ptr<float>();
  const float* color_in = color_logits.data_ptr<float>();
  const float* qty_in = qty_logits.data_ptr<float>();

  torch::Tensor verb_out = apply_to_logits ? verb_logits.clone() : verb_logits;
  torch::Tensor node_out = apply_to_logits ? node_logits.clone() : node_logits;
  torch::Tensor color_out = apply_to_logits ? color_logits.clone() : color_logits;
  torch::Tensor qty_out = apply_to_logits ? qty_logits.clone() : qty_logits;

  float* verb_out_ptr = verb_out.data_ptr<float>();
  float* node_out_ptr = node_out.data_ptr<float>();
  float* color_out_ptr = color_out.data_ptr<float>();
  float* qty_out_ptr = qty_out.data_ptr<float>();

  auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
  torch::Tensor verb_mask = torch::empty({B, n_verbs}, bool_opts);
  torch::Tensor node_mask = torch::zeros({B, n_nodes}, bool_opts);
  torch::Tensor color_mask = torch::zeros({B, n_colors}, bool_opts);
  torch::Tensor qty_mask = torch::zeros({B, Q}, bool_opts);

  auto* verb_mask_ptr = verb_mask.data_ptr<bool>();
  auto* node_mask_ptr = node_mask.data_ptr<bool>();
  auto* color_mask_ptr = color_mask.data_ptr<bool>();
  auto* qty_mask_ptr = qty_mask.data_ptr<bool>();

  const float big_neg_f = static_cast<float>(big_neg);
  const int color_yellow = static_cast<int>(std::min<int64_t>(1, n_colors - 1));

  const bool flip_enable = can_flip && (kVerbFlip < n_verbs);
  const bool steal_enable = allow_steal && (kVerbSteal < n_verbs);

#pragma omp parallel for if (B > 64)
  for (int64_t b = 0; b < B; ++b) {
    bool* vmask = verb_mask_ptr + b * n_verbs;
    bool* nmask = node_mask_ptr + b * n_nodes;
    bool* cmask = color_mask_ptr + b * n_colors;
    bool* qmask = qty_mask_ptr + b * Q;

    std::fill(vmask, vmask + n_verbs, false);
    std::fill(nmask, nmask + n_nodes, false);
    std::fill(cmask, cmask + n_colors, false);
    std::fill(qmask, qmask + Q, false);

    const int64_t blue_node_id = blue_reader.get<int64_t>(b);
    RowData row = compute_row_data(
        b,
        cap_reader,
        inv_sum_reader,
        nest_reader,
        inv_reader,
        pick_reader,
        pan_reader,
        pan_room_reader,
        n_colors,
        n_pickups,
        n_pantries,
        color_yellow,
        nest_cap,
        nest_blue_id,
        allow_steal,
        can_flip,
        n_verbs);

    if (kVerbPick < n_verbs) {
      vmask[kVerbPick] = row.pick_valid;
    }
    if (kVerbPlace < n_verbs) {
      vmask[kVerbPlace] = row.place_valid;
    }
    if (kVerbFlip < n_verbs) {
      vmask[kVerbFlip] = row.flip_valid;
    }
    if (kVerbSteal < n_verbs) {
      vmask[kVerbSteal] = row.steal_valid;
    }

    int32_t verb_choice = -1;
    if (have_verb_sel) {
      int64_t val = verb_sel_ptr[b];
      if (val >= 0 && val < n_verbs) {
        verb_choice = static_cast<int32_t>(val);
      }
    }

    int64_t node_choice = -1;
    if (have_node_sel) {
      int64_t val = node_sel_ptr[b];
      if (val >= 0 && val < n_nodes) {
        node_choice = val;
      }
    }

    int32_t color_choice = -1;
    if (have_color_sel) {
      int64_t val = color_sel_ptr[b];
      if (val >= 0 && val < n_colors) {
        color_choice = static_cast<int32_t>(val);
      }
    }

    const bool node_required =
        (verb_choice == kVerbPick) || (verb_choice == kVerbPlace) || (verb_choice == kVerbSteal);
    const bool color_required =
        (verb_choice == kVerbPick) || (verb_choice == kVerbPlace) ||
        (verb_choice == kVerbFlip) || (verb_choice == kVerbSteal);
    const bool qty_required = color_required;

    if (node_required) {
      for (int64_t node = 0; node < n_nodes; ++node) {
        bool valid = false;
        const int64_t pickup_local = pickup_idx_ptr[node];
        const int64_t pantry_local = pantry_idx_ptr[node];
        if (verb_choice == kVerbPick && row.cap > 0 && pickup_local >= 0 && pickup_local < n_pickups) {
          valid = row.pickup_best[pickup_local] > 0;
        }
        if (verb_choice == kVerbPlace && row.total_inv > 0) {
          if (pantry_local >= 0 && pantry_local < n_pantries) {
            valid = row.pantry_room[pantry_local] > 0;
          }
          if (!valid && row.nest_cap_left > 0 && nest_blue_id >= 0) {
            if (node == nest_blue_id) {
              valid = true;
            }
          }
        }
        if (verb_choice == kVerbFlip && row.flip_valid) {
          if (flip_node_all) {
            valid = true;
          } else if (node == blue_node_id) {
            valid = true;
          }
        }
        if (verb_choice == kVerbSteal && row.cap > 0 && pantry_local >= 0 && pantry_local < n_pantries) {
          valid = row.pantry_yellow[pantry_local] > 0;
        }
        nmask[node] = valid;
      }

      int64_t fallback_node = ensure_mask_has_true(
          nmask, n_nodes, node_in + b * n_nodes, 0);
      (void)fallback_node;
      if (apply_to_logits) {
        apply_big_neg(node_out_ptr + b * n_nodes, nmask, n_nodes, big_neg_f);
      }
    }

    if (color_required) {
      if (verb_choice == kVerbPick) {
        const int64_t pickup_local =
            (node_choice >= 0 && node_choice < n_nodes) ? pickup_idx_ptr[node_choice] : -1;
        if (pickup_local >= 0 && pickup_local < n_pickups && row.cap > 0) {
          for (int64_t c = 0; c < n_colors; ++c) {
            cmask[c] = row.pickup_stock[pickup_local * n_colors + c] > 0;
          }
        }
      } else if (verb_choice == kVerbPlace) {
        const int64_t pantry_local =
            (node_choice >= 0 && node_choice < n_nodes) ? pantry_idx_ptr[node_choice] : -1;
        if (pantry_local >= 0 && pantry_local < n_pantries) {
          int32_t room = row.pantry_room[pantry_local];
          if (room > 0) {
            for (int64_t c = 0; c < n_colors; ++c) {
              cmask[c] = row.inv_colors[c] > 0;
            }
          }
        } else if (row.nest_cap_left > 0 && nest_blue_id >= 0 && node_choice == nest_blue_id) {
          for (int64_t c = 0; c < n_colors; ++c) {
            cmask[c] = row.inv_colors[c] > 0;
          }
        }
      } else if (verb_choice == kVerbFlip) {
        if (row.total_inv > 0) {
          for (int64_t c = 0; c < n_colors; ++c) {
            int32_t other = row.total_inv - row.inv_colors[c];
            cmask[c] = other > 0;
          }
        }
      } else if (verb_choice == kVerbSteal) {
        const int64_t pantry_local =
            (node_choice >= 0 && node_choice < n_nodes) ? pantry_idx_ptr[node_choice] : -1;
        if (pantry_local >= 0 && pantry_local < n_pantries && row.cap > 0) {
          for (int64_t c = 0; c < n_colors; ++c) {
            cmask[c] = row.pantry_stock[pantry_local * n_colors + c] > 0;
          }
        }
      }

      ensure_mask_has_true(
          cmask, n_colors, color_in + b * n_colors, 0);
      if (apply_to_logits) {
        apply_big_neg(color_out_ptr + b * n_colors, cmask, n_colors, big_neg_f);
      }
    }

    if (qty_required && color_choice >= 0 && color_choice < n_colors) {
      qmask[0] = 0;
      int32_t limit = 0;
      if (verb_choice == kVerbPick) {
        const int64_t pickup_local =
            (node_choice >= 0 && node_choice < n_nodes) ? pickup_idx_ptr[node_choice] : -1;
        if (pickup_local >= 0 && pickup_local < n_pickups && row.cap > 0) {
          int32_t stock = row.pickup_stock[pickup_local * n_colors + color_choice];
          limit = std::min(stock, row.cap);
        }
      } else if (verb_choice == kVerbPlace) {
        const int64_t pantry_local =
            (node_choice >= 0 && node_choice < n_nodes) ? pantry_idx_ptr[node_choice] : -1;
        if (pantry_local >= 0 && pantry_local < n_pantries) {
          int32_t room = row.pantry_room[pantry_local];
          int32_t inv_val = row.inv_colors[color_choice];
          limit = std::min(room, inv_val);
        } else if (row.nest_cap_left > 0 && nest_blue_id >= 0 && node_choice == nest_blue_id) {
          int32_t inv_val = row.inv_colors[color_choice];
          limit = std::min(row.nest_cap_left, inv_val);
        }
      } else if (verb_choice == kVerbFlip) {
        int32_t inv_val = row.inv_colors[color_choice];
        int32_t other = row.total_inv - inv_val;
        limit = other;
      } else if (verb_choice == kVerbSteal) {
        const int64_t pantry_local =
            (node_choice >= 0 && node_choice < n_nodes) ? pantry_idx_ptr[node_choice] : -1;
        if (pantry_local >= 0 && pantry_local < n_pantries && row.cap > 0) {
          int32_t stock = row.pantry_stock[pantry_local * n_colors + color_choice];
          limit = std::min(stock, row.cap);
        }
      }

      limit = clamp_positive(limit);
      if (limit > Q - 1) {
        limit = static_cast<int32_t>(Q - 1);
      }
      for (int32_t q = 1; q <= limit; ++q) {
        qmask[q] = 1;
      }

      if (Q > 1) {
        ensure_mask_has_true(
            qmask, Q, qty_in + b * Q, 1);
      }
      if (apply_to_logits) {
        float* qty_row = qty_out_ptr + b * Q;
        qty_row[0] = big_neg_f;
        for (int64_t q = 1; q < Q; ++q) {
          if (!qmask[q]) {
            qty_row[q] = big_neg_f;
          }
        }
      }
    } else if (apply_to_logits) {
      float* qty_row = qty_out_ptr + b * Q;
      qty_row[0] = big_neg_f;
      for (int64_t q = 1; q < Q; ++q) {
        qty_row[q] = big_neg_f;
      }
    }

    ensure_mask_has_true(
        vmask, n_verbs, verb_in + b * n_verbs, 0);
    if (apply_to_logits) {
      apply_big_neg(verb_out_ptr + b * n_verbs, vmask, n_verbs, big_neg_f);
    }
  }

  return std::make_tuple(
      verb_mask,
      node_mask,
      color_mask,
      qty_mask,
      verb_out,
      node_out,
      color_out,
      qty_out);
}

torch::Tensor qty_limit_given(
    const torch::Tensor& verb_sel,
    const torch::Tensor& node_sel,
    const torch::Tensor& color_sel,
    const torch::Tensor& inv_b,
    const torch::Tensor& cap_left,
    const torch::Tensor& pan,
    const torch::Tensor& pick,
    const torch::Tensor& pan_room,
    const torch::Tensor& nest_blue,
    const torch::Tensor& pantry_idx,
    const torch::Tensor& pickup_idx,
    int64_t nest_cap,
    int64_t nest_blue_id) {
  TORCH_CHECK(verb_sel.device().is_cpu(), "verb_sel must be on CPU");
  TORCH_CHECK(node_sel.device().is_cpu(), "node_sel must be on CPU");
  TORCH_CHECK(color_sel.device().is_cpu(), "color_sel must be on CPU");
  TORCH_CHECK(inv_b.device().is_cpu(), "inv_b must be on CPU");
  TORCH_CHECK(cap_left.device().is_cpu(), "cap_left must be on CPU");
  TORCH_CHECK(pan.device().is_cpu(), "pan must be on CPU");
  TORCH_CHECK(pick.device().is_cpu(), "pick must be on CPU");
  TORCH_CHECK(pan_room.device().is_cpu(), "pan_room must be on CPU");
  TORCH_CHECK(nest_blue.device().is_cpu(), "nest_blue must be on CPU");
  TORCH_CHECK(pantry_idx.device().is_cpu(), "pantry_idx must be on CPU");
  TORCH_CHECK(pickup_idx.device().is_cpu(), "pickup_idx must be on CPU");

  TORCH_CHECK(verb_sel.is_contiguous(), "verb_sel must be contiguous");
  TORCH_CHECK(node_sel.is_contiguous(), "node_sel must be contiguous");
  TORCH_CHECK(color_sel.is_contiguous(), "color_sel must be contiguous");
  TORCH_CHECK(inv_b.is_contiguous(), "inv_b must be contiguous");
  TORCH_CHECK(cap_left.is_contiguous(), "cap_left must be contiguous");
  TORCH_CHECK(pan.is_contiguous(), "pan must be contiguous");
  TORCH_CHECK(pick.is_contiguous(), "pick must be contiguous");
  TORCH_CHECK(pan_room.is_contiguous(), "pan_room must be contiguous");
  TORCH_CHECK(nest_blue.is_contiguous(), "nest_blue must be contiguous");
  TORCH_CHECK(pantry_idx.is_contiguous(), "pantry_idx must be contiguous");
  TORCH_CHECK(pickup_idx.is_contiguous(), "pickup_idx must be contiguous");

  TORCH_CHECK(verb_sel.dim() == 1, "verb_sel must be 1D");
  TORCH_CHECK(node_sel.dim() == 1 && node_sel.size(0) == verb_sel.size(0), "node_sel shape mismatch");
  TORCH_CHECK(color_sel.dim() == 1 && color_sel.size(0) == verb_sel.size(0), "color_sel shape mismatch");
  TORCH_CHECK(inv_b.dim() == 2 && inv_b.size(0) == verb_sel.size(0), "inv_b shape mismatch");
  TORCH_CHECK(cap_left.dim() == 1 && cap_left.size(0) == verb_sel.size(0), "cap_left shape mismatch");
  TORCH_CHECK(pan.dim() == 3 && pan.size(0) == verb_sel.size(0), "pan shape mismatch");
  TORCH_CHECK(pick.dim() == 3 && pick.size(0) == verb_sel.size(0), "pick shape mismatch");
  TORCH_CHECK(pan_room.dim() == 2 && pan_room.size(0) == verb_sel.size(0), "pan_room shape mismatch");
  TORCH_CHECK(nest_blue.dim() == 1 && nest_blue.size(0) == verb_sel.size(0), "nest_blue shape mismatch");

  const int64_t B = verb_sel.size(0);
  const int64_t n_colors = inv_b.size(1);
  const int64_t n_pantries = pan.size(1);
  const int64_t n_pickups = pick.size(1);

  Reader1D verb_reader(verb_sel);
  Reader1D node_reader(node_sel);
  Reader1D color_reader(color_sel);
  Reader2D inv_reader(inv_b);
  Reader1D cap_reader(cap_left);
  Reader3D pan_reader(pan);
  Reader3D pick_reader(pick);
  Reader2D pan_room_reader(pan_room);
  Reader1D nest_reader(nest_blue);

  const auto pantry_idx_ptr = pantry_idx.data_ptr<int64_t>();
  const auto pickup_idx_ptr = pickup_idx.data_ptr<int64_t>();

  auto out = torch::empty({B}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCPU));
  auto* out_ptr = out.data_ptr<int32_t>();

#pragma omp parallel for if (B > 64)
  for (int64_t b = 0; b < B; ++b) {
    const int32_t verb = verb_reader.get<int32_t>(b);
    const int64_t node = node_reader.get<int64_t>(b);
    const int32_t color = color_reader.get<int32_t>(b);
    int32_t cap = clamp_positive(cap_reader.get<int32_t>(b));
    int32_t limit = 0;

    if (verb == kVerbPick) {
      int32_t local = -1;
      if (node >= 0 && node < pickup_idx.size(0)) {
        local = pickup_idx_ptr[node];
      }
      if (local >= 0 && local < n_pickups && color >= 0 && color < n_colors) {
        int32_t stock = clamp_positive(pick_reader.get<int32_t>(b, local, color));
        limit = cap > 0 ? std::min(stock, cap) : stock;
      }
    } else if (verb == kVerbPlace) {
      bool handled = false;
      if (node == nest_blue_id && nest_cap > 0) {
        int32_t nest_count = clamp_positive(nest_reader.get<int32_t>(b));
        int64_t diff = nest_cap - static_cast<int64_t>(nest_count);
        int32_t nest_cap_left = diff > 0 ? static_cast<int32_t>(diff) : 0;
        if (nest_cap_left > 0 && color >= 0 && color < n_colors) {
          int32_t inv_val = clamp_positive(inv_reader.get<int32_t>(b, color));
          limit = std::min(nest_cap_left, inv_val);
        }
        handled = true;
      }
      if (!handled && node >= 0 && node < pantry_idx.size(0)) {
        int32_t local = static_cast<int32_t>(pantry_idx_ptr[node]);
        if (local >= 0 && local < n_pantries && color >= 0 && color < n_colors) {
          int32_t room = clamp_positive(pan_room_reader.get<int32_t>(b, local));
          int32_t inv_val = clamp_positive(inv_reader.get<int32_t>(b, color));
          limit = std::min(room, inv_val);
        }
      }
    } else if (verb == kVerbFlip) {
      int32_t total = 0;
      for (int64_t c = 0; c < n_colors; ++c) {
        total += clamp_positive(inv_reader.get<int32_t>(b, c));
      }
      if (color >= 0 && color < n_colors) {
        int32_t inv_val = clamp_positive(inv_reader.get<int32_t>(b, color));
        limit = total - inv_val;
      }
    } else if (verb == kVerbSteal) {
      if (node >= 0 && node < pantry_idx.size(0)) {
        int32_t local = static_cast<int32_t>(pantry_idx_ptr[node]);
        if (local >= 0 && local < n_pantries && color >= 0 && color < n_colors) {
          int32_t stock = clamp_positive(pan_reader.get<int32_t>(b, local, color));
          limit = cap > 0 ? std::min(stock, cap) : stock;
        }
      }
    }
    out_ptr[b] = clamp_positive(limit);
  }

  return out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_verb(
    const torch::Tensor& verb_logits,
    const torch::Tensor& blue_node,
    const torch::Tensor& inv_b,
    const torch::Tensor& inv_sum,
    const torch::Tensor& cap_left,
    const torch::Tensor& pan,
    const torch::Tensor& pick,
    const torch::Tensor& pan_room,
    const torch::Tensor& pan_sum,
    const torch::Tensor& pick_sum,
    const torch::Tensor& nest_blue,
    const torch::Tensor& pantry_idx,
    const torch::Tensor& pickup_idx,
    const torch::Tensor& pantry_nodes,
    const torch::Tensor& pickup_nodes,
    int64_t n_verbs,
    int64_t n_nodes,
    int64_t n_colors,
    int64_t Q,
    int64_t capacity,
    int64_t pantry_cap,
    int64_t nest_cap,
    int64_t nest_blue_id,
    bool allow_steal,
    bool can_flip,
    double big_neg,
    bool flip_node_all) {
  (void)pan_sum;
  (void)pick_sum;
  (void)capacity;
  (void)pantry_cap;
  (void)Q;
  (void)n_nodes;
  (void)flip_node_all;
  TORCH_CHECK(verb_logits.device().is_cpu(), "verb_logits must be on CPU");
  TORCH_CHECK(verb_logits.is_contiguous(), "verb_logits must be contiguous");
  TORCH_CHECK(verb_logits.dim() == 2, "verb_logits must be 2D");
  TORCH_CHECK(verb_logits.scalar_type() == torch::kFloat32, "verb_logits must be float32");

  const int64_t B = verb_logits.size(0);
  TORCH_CHECK(verb_logits.size(1) == n_verbs, "verb_logits size(1) mismatch");
  TORCH_CHECK(blue_node.dim() == 1 && blue_node.size(0) == B, "blue_node shape mismatch");
  TORCH_CHECK(inv_b.dim() == 2 && inv_b.size(0) == B && inv_b.size(1) == n_colors, "inv_b shape mismatch");
  TORCH_CHECK(inv_sum.dim() == 1 && inv_sum.size(0) == B, "inv_sum shape mismatch");
  TORCH_CHECK(cap_left.dim() == 1 && cap_left.size(0) == B, "cap_left shape mismatch");
  TORCH_CHECK(pan.dim() == 3 && pan.size(0) == B && pan.size(2) == n_colors, "pan shape mismatch");
  TORCH_CHECK(pick.dim() == 3 && pick.size(0) == B && pick.size(2) == n_colors, "pick shape mismatch");
  TORCH_CHECK(pan_room.dim() == 2 && pan_room.size(0) == B && pan_room.size(1) == pan.size(1), "pan_room shape mismatch");
  TORCH_CHECK(nest_blue.dim() == 1 && nest_blue.size(0) == B, "nest_blue shape mismatch");
  TORCH_CHECK(pantry_idx.dim() == 1 && pantry_idx.size(0) == n_nodes, "pantry_idx shape mismatch");
  TORCH_CHECK(pickup_idx.dim() == 1 && pickup_idx.size(0) == n_nodes, "pickup_idx shape mismatch");

  Reader1D cap_reader(cap_left);
  Reader1D inv_sum_reader(inv_sum);
  Reader1D nest_reader(nest_blue);
  Reader2D inv_reader(inv_b);
  Reader3D pick_reader(pick);
  Reader3D pan_reader(pan);
  Reader2D pan_room_reader(pan_room);

  const float big_neg_f = static_cast<float>(big_neg);
  const int color_yellow = static_cast<int>(std::min<int64_t>(1, n_colors - 1));

  torch::Tensor masked_logits = verb_logits.clone();
  auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
  torch::Tensor mask = torch::empty({B, n_verbs}, bool_opts);

  float* masked_ptr = masked_logits.data_ptr<float>();
  const float* logits_ptr = verb_logits.data_ptr<float>();
  auto* mask_ptr = mask.data_ptr<bool>();

#pragma omp parallel for if (B > 64)
  for (int64_t b = 0; b < B; ++b) {
    bool* row_mask = mask_ptr + b * n_verbs;
    std::fill(row_mask, row_mask + n_verbs, false);
    RowData row = compute_row_data(
        b,
        cap_reader,
        inv_sum_reader,
        nest_reader,
        inv_reader,
        pick_reader,
        pan_reader,
        pan_room_reader,
        n_colors,
        pick.size(1),
        pan.size(1),
        color_yellow,
        nest_cap,
        nest_blue_id,
        allow_steal,
        can_flip,
        n_verbs);

    if (kVerbPick < n_verbs) {
      row_mask[kVerbPick] = row.pick_valid;
    }
    if (kVerbPlace < n_verbs) {
      row_mask[kVerbPlace] = row.place_valid;
    }
    if (kVerbFlip < n_verbs) {
      row_mask[kVerbFlip] = row.flip_valid;
    }
    if (kVerbSteal < n_verbs) {
      row_mask[kVerbSteal] = row.steal_valid;
    }

    ensure_mask_has_true(row_mask, n_verbs, logits_ptr + b * n_verbs, 0);
    apply_big_neg(masked_ptr + b * n_verbs, row_mask, n_verbs, big_neg_f);
  }

  auto uniform = torch::empty_like(masked_logits).uniform_(1e-6f, 1.0f);
  auto gumbel = -torch::log(-torch::log(uniform));
  auto noisy = masked_logits + gumbel;
  auto max_pair = noisy.max(1);
  torch::Tensor indices = std::get<1>(max_pair);
  torch::Tensor log_soft = torch::log_softmax(masked_logits, 1);
  torch::Tensor logp = log_soft.gather(1, indices.unsqueeze(1)).squeeze(1);
  torch::Tensor prob = torch::softmax(masked_logits, 1);
  torch::Tensor entropy = (-prob * log_soft).sum(1);

  return std::make_tuple(indices, logp, entropy);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_node(
    const torch::Tensor& node_logits,
    const torch::Tensor& blue_node,
    const torch::Tensor& inv_b,
    const torch::Tensor& inv_sum,
    const torch::Tensor& cap_left,
    const torch::Tensor& pan,
    const torch::Tensor& pick,
    const torch::Tensor& pan_room,
    const torch::Tensor& pan_sum,
    const torch::Tensor& pick_sum,
    const torch::Tensor& nest_blue,
    const torch::Tensor& pantry_idx,
    const torch::Tensor& pickup_idx,
    const torch::Tensor& pantry_nodes,
    const torch::Tensor& pickup_nodes,
    const torch::Tensor& verb_idx,
    int64_t n_verbs,
    int64_t n_nodes,
    int64_t n_colors,
    int64_t Q,
    int64_t capacity,
    int64_t pantry_cap,
    int64_t nest_cap,
    int64_t nest_blue_id,
    bool allow_steal,
    bool can_flip,
    double big_neg,
    bool flip_node_all) {
  (void)pan_sum;
  (void)pick_sum;
  (void)capacity;
  (void)pantry_cap;
  (void)pantry_nodes;
  (void)pickup_nodes;
  (void)Q;
  TORCH_CHECK(node_logits.device().is_cpu(), "node_logits must be on CPU");
  TORCH_CHECK(node_logits.is_contiguous(), "node_logits must be contiguous");
  TORCH_CHECK(node_logits.dim() == 2, "node_logits must be 2D");
  TORCH_CHECK(node_logits.scalar_type() == torch::kFloat32, "node_logits must be float32");
  TORCH_CHECK(verb_idx.device().is_cpu(), "verb_idx must be on CPU");
  TORCH_CHECK(verb_idx.is_contiguous(), "verb_idx must be contiguous");
  TORCH_CHECK(verb_idx.dim() == 1, "verb_idx must be 1D");

  const int64_t B = node_logits.size(0);
  TORCH_CHECK(node_logits.size(1) == n_nodes, "node_logits size(1) mismatch");
  TORCH_CHECK(verb_idx.size(0) == B, "verb_idx batch mismatch");
  TORCH_CHECK(blue_node.dim() == 1 && blue_node.size(0) == B, "blue_node shape mismatch");
  TORCH_CHECK(inv_b.dim() == 2 && inv_b.size(0) == B && inv_b.size(1) == n_colors, "inv_b shape mismatch");
  TORCH_CHECK(inv_sum.dim() == 1 && inv_sum.size(0) == B, "inv_sum shape mismatch");
  TORCH_CHECK(cap_left.dim() == 1 && cap_left.size(0) == B, "cap_left shape mismatch");
  TORCH_CHECK(pan.dim() == 3 && pan.size(0) == B && pan.size(2) == n_colors, "pan shape mismatch");
  TORCH_CHECK(pick.dim() == 3 && pick.size(0) == B && pick.size(2) == n_colors, "pick shape mismatch");
  TORCH_CHECK(pan_room.dim() == 2 && pan_room.size(0) == B && pan_room.size(1) == pan.size(1), "pan_room shape mismatch");
  TORCH_CHECK(nest_blue.dim() == 1 && nest_blue.size(0) == B, "nest_blue shape mismatch");
  TORCH_CHECK(pantry_idx.dim() == 1 && pantry_idx.size(0) == n_nodes, "pantry_idx shape mismatch");
  TORCH_CHECK(pickup_idx.dim() == 1 && pickup_idx.size(0) == n_nodes, "pickup_idx shape mismatch");

  Reader1D cap_reader(cap_left);
  Reader1D inv_sum_reader(inv_sum);
  Reader1D nest_reader(nest_blue);
  Reader2D inv_reader(inv_b);
  Reader3D pick_reader(pick);
  Reader3D pan_reader(pan);
  Reader2D pan_room_reader(pan_room);
  Reader1D blue_reader(blue_node);
  Reader1D verb_reader(verb_idx);

  const auto pantry_idx_ptr = pantry_idx.data_ptr<int64_t>();
  const auto pickup_idx_ptr = pickup_idx.data_ptr<int64_t>();

  const float big_neg_f = static_cast<float>(big_neg);
  const int color_yellow = static_cast<int>(std::min<int64_t>(1, n_colors - 1));

  torch::Tensor masked_logits = node_logits.clone();
  auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
  torch::Tensor mask = torch::empty({B, n_nodes}, bool_opts);

  float* masked_ptr = masked_logits.data_ptr<float>();
  const float* logits_ptr = node_logits.data_ptr<float>();
  auto* mask_ptr = mask.data_ptr<bool>();

  std::vector<int64_t> forced_indices(B, -1);
  std::vector<bool> active(B, true);

#pragma omp parallel for if (B > 64)
  for (int64_t b = 0; b < B; ++b) {
    bool* row_mask = mask_ptr + b * n_nodes;
    std::fill(row_mask, row_mask + n_nodes, false);
    RowData row = compute_row_data(
        b,
        cap_reader,
        inv_sum_reader,
        nest_reader,
        inv_reader,
        pick_reader,
        pan_reader,
        pan_room_reader,
        n_colors,
        pick.size(1),
        pan.size(1),
        color_yellow,
        nest_cap,
        nest_blue_id,
        allow_steal,
        can_flip,
        n_verbs);

    int64_t verb_val = verb_reader.get<int64_t>(b);
    const int64_t blue_node_id = blue_reader.get<int64_t>(b);
    if (verb_val < 0 || verb_val >= n_verbs) {
      active[b] = false;
      int64_t forced = blue_node_id;
      if (forced < 0 || forced >= n_nodes) {
        forced = 0;
      }
      forced_indices[b] = forced;
      row_mask[forced] = true;
      continue;
    }

    switch (verb_val) {
      case kVerbPick: {
        for (int64_t node = 0; node < n_nodes; ++node) {
          const int64_t pickup_local = pickup_idx_ptr[node];
          bool valid = row.cap > 0 && pickup_local >= 0 && pickup_local < pick.size(1) &&
              row.pickup_best[pickup_local] > 0;
          row_mask[node] = valid;
        }
        break;
      }
      case kVerbPlace: {
        for (int64_t node = 0; node < n_nodes; ++node) {
          const int64_t pantry_local = pantry_idx_ptr[node];
          bool valid = false;
          if (pantry_local >= 0 && pantry_local < pan.size(1)) {
            valid = row.pantry_room[pantry_local] > 0 && row.total_inv > 0;
          }
          if (!valid && row.nest_cap_left > 0 && nest_blue_id >= 0 && node == nest_blue_id) {
            valid = true;
          }
          row_mask[node] = valid;
        }
        break;
      }
      case kVerbFlip: {
        if (row.flip_valid) {
          if (flip_node_all) {
            std::fill(row_mask, row_mask + n_nodes, true);
          } else if (blue_node_id >= 0 && blue_node_id < n_nodes) {
            row_mask[blue_node_id] = true;
          }
        } else if (blue_node_id >= 0 && blue_node_id < n_nodes) {
          row_mask[blue_node_id] = true;
        }
        break;
      }
      case kVerbSteal: {
        for (int64_t node = 0; node < n_nodes; ++node) {
          const int64_t pantry_local = pantry_idx_ptr[node];
          bool valid = row.cap > 0 && pantry_local >= 0 && pantry_local < pan.size(1) &&
              row.pantry_yellow[pantry_local] > 0;
          row_mask[node] = valid;
        }
        break;
      }
      default: {
        int64_t forced = blue_node_id;
        if (forced < 0 || forced >= n_nodes) {
          forced = 0;
        }
        row_mask[forced] = true;
        break;
      }
    }

    ensure_mask_has_true(row_mask, n_nodes, logits_ptr + b * n_nodes, 0);
    apply_big_neg(masked_ptr + b * n_nodes, row_mask, n_nodes, big_neg_f);
  }

  auto uniform = torch::empty_like(masked_logits).uniform_(1e-6f, 1.0f);
  auto gumbel = -torch::log(-torch::log(uniform));
  auto noisy = masked_logits + gumbel;
  auto max_pair = noisy.max(1);
  torch::Tensor indices = std::get<1>(max_pair);
  torch::Tensor log_soft = torch::log_softmax(masked_logits, 1);
  torch::Tensor logp = log_soft.gather(1, indices.unsqueeze(1)).squeeze(1);
  torch::Tensor prob = torch::softmax(masked_logits, 1);
  torch::Tensor entropy = (-prob * log_soft).sum(1);

  auto* indices_ptr = indices.data_ptr<int64_t>();
  auto* logp_ptr = logp.data_ptr<float>();
  auto* entropy_ptr = entropy.data_ptr<float>();
  for (int64_t b = 0; b < B; ++b) {
    if (!active[b]) {
      int64_t forced = forced_indices[b];
      if (forced < 0 || forced >= n_nodes) {
        forced = 0;
      }
      indices_ptr[b] = forced;
      logp_ptr[b] = 0.0f;
      entropy_ptr[b] = 0.0f;
    }
  }

  return std::make_tuple(indices, logp, entropy);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_color(
    const torch::Tensor& color_logits,
    const torch::Tensor& blue_node,
    const torch::Tensor& inv_b,
    const torch::Tensor& inv_sum,
    const torch::Tensor& cap_left,
    const torch::Tensor& pan,
    const torch::Tensor& pick,
    const torch::Tensor& pan_room,
    const torch::Tensor& pan_sum,
    const torch::Tensor& pick_sum,
    const torch::Tensor& nest_blue,
    const torch::Tensor& pantry_idx,
    const torch::Tensor& pickup_idx,
    const torch::Tensor& pantry_nodes,
    const torch::Tensor& pickup_nodes,
    const torch::Tensor& verb_idx,
    const torch::Tensor& node_idx,
    int64_t n_verbs,
    int64_t n_nodes,
    int64_t n_colors,
    int64_t Q,
    int64_t capacity,
    int64_t pantry_cap,
    int64_t nest_cap,
    int64_t nest_blue_id,
    bool allow_steal,
    bool can_flip,
    double big_neg,
    bool flip_node_all) {
  (void)pan_sum;
  (void)pick_sum;
  (void)capacity;
  (void)pantry_cap;
  (void)pantry_nodes;
  (void)pickup_nodes;
  (void)Q;
  (void)flip_node_all;
  TORCH_CHECK(color_logits.device().is_cpu(), "color_logits must be on CPU");
  TORCH_CHECK(color_logits.is_contiguous(), "color_logits must be contiguous");
  TORCH_CHECK(color_logits.dim() == 2, "color_logits must be 2D");
  TORCH_CHECK(color_logits.scalar_type() == torch::kFloat32, "color_logits must be float32");
  TORCH_CHECK(verb_idx.device().is_cpu() && node_idx.device().is_cpu(), "indices must be on CPU");
  TORCH_CHECK(verb_idx.is_contiguous() && node_idx.is_contiguous(), "indices must be contiguous");
  TORCH_CHECK(verb_idx.dim() == 1 && node_idx.dim() == 1, "indices must be 1D");

  const int64_t B = color_logits.size(0);
  TORCH_CHECK(color_logits.size(1) == n_colors, "color_logits size(1) mismatch");
  TORCH_CHECK(verb_idx.size(0) == B && node_idx.size(0) == B, "index batch mismatch");
  TORCH_CHECK(blue_node.dim() == 1 && blue_node.size(0) == B, "blue_node shape mismatch");
  TORCH_CHECK(inv_b.dim() == 2 && inv_b.size(0) == B && inv_b.size(1) == n_colors, "inv_b shape mismatch");
  TORCH_CHECK(inv_sum.dim() == 1 && inv_sum.size(0) == B, "inv_sum shape mismatch");
  TORCH_CHECK(cap_left.dim() == 1 && cap_left.size(0) == B, "cap_left shape mismatch");
  TORCH_CHECK(pan.dim() == 3 && pan.size(0) == B && pan.size(2) == n_colors, "pan shape mismatch");
  TORCH_CHECK(pick.dim() == 3 && pick.size(0) == B && pick.size(2) == n_colors, "pick shape mismatch");
  TORCH_CHECK(pan_room.dim() == 2 && pan_room.size(0) == B && pan_room.size(1) == pan.size(1), "pan_room shape mismatch");
  TORCH_CHECK(nest_blue.dim() == 1 && nest_blue.size(0) == B, "nest_blue shape mismatch");
  TORCH_CHECK(pantry_idx.dim() == 1 && pantry_idx.size(0) == n_nodes, "pantry_idx shape mismatch");
  TORCH_CHECK(pickup_idx.dim() == 1 && pickup_idx.size(0) == n_nodes, "pickup_idx shape mismatch");

  Reader1D cap_reader(cap_left);
  Reader1D inv_sum_reader(inv_sum);
  Reader1D nest_reader(nest_blue);
  Reader2D inv_reader(inv_b);
  Reader3D pick_reader(pick);
  Reader3D pan_reader(pan);
  Reader2D pan_room_reader(pan_room);
  Reader1D verb_reader(verb_idx);
  Reader1D node_reader(node_idx);

  const auto pantry_idx_ptr = pantry_idx.data_ptr<int64_t>();
  const auto pickup_idx_ptr = pickup_idx.data_ptr<int64_t>();

  const float big_neg_f = static_cast<float>(big_neg);
  const int color_yellow = static_cast<int>(std::min<int64_t>(1, n_colors - 1));

  torch::Tensor masked_logits = color_logits.clone();
  auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
  torch::Tensor mask = torch::empty({B, n_colors}, bool_opts);

  float* masked_ptr = masked_logits.data_ptr<float>();
  const float* logits_ptr = color_logits.data_ptr<float>();
  auto* mask_ptr = mask.data_ptr<bool>();

  std::vector<int64_t> forced_indices(B, -1);
  std::vector<bool> active(B, true);

#pragma omp parallel for if (B > 64)
  for (int64_t b = 0; b < B; ++b) {
    bool* row_mask = mask_ptr + b * n_colors;
    std::fill(row_mask, row_mask + n_colors, false);
    RowData row = compute_row_data(
        b,
        cap_reader,
        inv_sum_reader,
        nest_reader,
        inv_reader,
        pick_reader,
        pan_reader,
        pan_room_reader,
        n_colors,
        pick.size(1),
        pan.size(1),
        color_yellow,
        nest_cap,
        nest_blue_id,
        allow_steal,
        can_flip,
        n_verbs);

    int64_t verb_val = verb_reader.get<int64_t>(b);
    int64_t node_val = node_reader.get<int64_t>(b);
    if (verb_val < 0 || verb_val >= n_verbs || node_val < 0 || node_val >= n_nodes) {
      active[b] = false;
      forced_indices[b] = 0;
      row_mask[0] = true;
      continue;
    }

    if (verb_val == kVerbPick) {
      const int64_t pickup_local = pickup_idx_ptr[node_val];
      if (pickup_local >= 0 && pickup_local < pick.size(1) && row.cap > 0) {
        for (int64_t c = 0; c < n_colors; ++c) {
          row_mask[c] = row.pickup_stock[pickup_local * n_colors + c] > 0;
        }
      }
    } else if (verb_val == kVerbPlace) {
      const int64_t pantry_local = pantry_idx_ptr[node_val];
      if (pantry_local >= 0 && pantry_local < pan.size(1)) {
        int32_t room = row.pantry_room[pantry_local];
        if (room > 0) {
          for (int64_t c = 0; c < n_colors; ++c) {
            row_mask[c] = row.inv_colors[c] > 0;
          }
        }
      } else if (row.nest_cap_left > 0 && nest_blue_id >= 0 && node_val == nest_blue_id) {
        for (int64_t c = 0; c < n_colors; ++c) {
          row_mask[c] = row.inv_colors[c] > 0;
        }
      }
    } else if (verb_val == kVerbFlip) {
      if (row.total_inv > 0) {
        for (int64_t c = 0; c < n_colors; ++c) {
          int32_t other = row.total_inv - row.inv_colors[c];
          row_mask[c] = other > 0;
        }
      }
    } else if (verb_val == kVerbSteal) {
      const int64_t pantry_local = pantry_idx_ptr[node_val];
      if (pantry_local >= 0 && pantry_local < pan.size(1) && row.cap > 0) {
        for (int64_t c = 0; c < n_colors; ++c) {
          row_mask[c] = row.pantry_stock[pantry_local * n_colors + c] > 0;
        }
      }
    }

    ensure_mask_has_true(row_mask, n_colors, logits_ptr + b * n_colors, 0);
    apply_big_neg(masked_ptr + b * n_colors, row_mask, n_colors, big_neg_f);
  }

  auto uniform = torch::empty_like(masked_logits).uniform_(1e-6f, 1.0f);
  auto gumbel = -torch::log(-torch::log(uniform));
  auto noisy = masked_logits + gumbel;
  auto max_pair = noisy.max(1);
  torch::Tensor indices = std::get<1>(max_pair);
  torch::Tensor log_soft = torch::log_softmax(masked_logits, 1);
  torch::Tensor logp = log_soft.gather(1, indices.unsqueeze(1)).squeeze(1);
  torch::Tensor prob = torch::softmax(masked_logits, 1);
  torch::Tensor entropy = (-prob * log_soft).sum(1);

  auto* indices_ptr = indices.data_ptr<int64_t>();
  auto* logp_ptr = logp.data_ptr<float>();
  auto* entropy_ptr = entropy.data_ptr<float>();
  for (int64_t b = 0; b < B; ++b) {
    if (!active[b]) {
      indices_ptr[b] = forced_indices[b];
      logp_ptr[b] = 0.0f;
      entropy_ptr[b] = 0.0f;
    }
  }

  return std::make_tuple(indices, logp, entropy);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sample_qty(
    const torch::Tensor& qty_logits,
    const torch::Tensor& blue_node,
    const torch::Tensor& inv_b,
    const torch::Tensor& inv_sum,
    const torch::Tensor& cap_left,
    const torch::Tensor& pan,
    const torch::Tensor& pick,
    const torch::Tensor& pan_room,
    const torch::Tensor& pan_sum,
    const torch::Tensor& pick_sum,
    const torch::Tensor& nest_blue,
    const torch::Tensor& pantry_idx,
    const torch::Tensor& pickup_idx,
    const torch::Tensor& pantry_nodes,
    const torch::Tensor& pickup_nodes,
    const torch::Tensor& verb_idx,
    const torch::Tensor& node_idx,
    const torch::Tensor& color_idx,
    int64_t n_verbs,
    int64_t n_nodes,
    int64_t n_colors,
    int64_t Q,
    int64_t capacity,
    int64_t pantry_cap,
    int64_t nest_cap,
    int64_t nest_blue_id,
    bool allow_steal,
    bool can_flip,
    double big_neg,
    bool flip_node_all) {
  (void)pan_sum;
  (void)pick_sum;
  (void)capacity;
  (void)pantry_cap;
  (void)pantry_nodes;
  (void)pickup_nodes;
  (void)flip_node_all;
  TORCH_CHECK(qty_logits.device().is_cpu(), "qty_logits must be on CPU");
  TORCH_CHECK(qty_logits.is_contiguous(), "qty_logits must be contiguous");
  TORCH_CHECK(qty_logits.dim() == 2, "qty_logits must be 2D");
  TORCH_CHECK(qty_logits.scalar_type() == torch::kFloat32, "qty_logits must be float32");
  TORCH_CHECK(verb_idx.device().is_cpu() && node_idx.device().is_cpu() && color_idx.device().is_cpu(),
              "indices must be on CPU");
  TORCH_CHECK(verb_idx.is_contiguous() && node_idx.is_contiguous() && color_idx.is_contiguous(),
              "indices must be contiguous");
  TORCH_CHECK(verb_idx.dim() == 1 && node_idx.dim() == 1 && color_idx.dim() == 1, "indices must be 1D");

  const int64_t B = qty_logits.size(0);
  TORCH_CHECK(qty_logits.size(1) == Q, "qty_logits size(1) mismatch");
  TORCH_CHECK(verb_idx.size(0) == B && node_idx.size(0) == B && color_idx.size(0) == B, "index batch mismatch");
  TORCH_CHECK(inv_b.dim() == 2 && inv_b.size(0) == B && inv_b.size(1) == n_colors, "inv_b shape mismatch");
  TORCH_CHECK(inv_sum.dim() == 1 && inv_sum.size(0) == B, "inv_sum shape mismatch");
  TORCH_CHECK(cap_left.dim() == 1 && cap_left.size(0) == B, "cap_left shape mismatch");
  TORCH_CHECK(pan.dim() == 3 && pan.size(0) == B && pan.size(2) == n_colors, "pan shape mismatch");
  TORCH_CHECK(pick.dim() == 3 && pick.size(0) == B && pick.size(2) == n_colors, "pick shape mismatch");
  TORCH_CHECK(pan_room.dim() == 2 && pan_room.size(0) == B && pan_room.size(1) == pan.size(1), "pan_room shape mismatch");
  TORCH_CHECK(nest_blue.dim() == 1 && nest_blue.size(0) == B, "nest_blue shape mismatch");
  TORCH_CHECK(pantry_idx.dim() == 1 && pantry_idx.size(0) == n_nodes, "pantry_idx shape mismatch");
  TORCH_CHECK(pickup_idx.dim() == 1 && pickup_idx.size(0) == n_nodes, "pickup_idx shape mismatch");

  Reader1D cap_reader(cap_left);
  Reader1D inv_sum_reader(inv_sum);
  Reader1D nest_reader(nest_blue);
  Reader2D inv_reader(inv_b);
  Reader3D pick_reader(pick);
  Reader3D pan_reader(pan);
  Reader2D pan_room_reader(pan_room);
  Reader1D verb_reader(verb_idx);
  Reader1D node_reader(node_idx);
  Reader1D color_reader(color_idx);

  const auto pantry_idx_ptr = pantry_idx.data_ptr<int64_t>();
  const auto pickup_idx_ptr = pickup_idx.data_ptr<int64_t>();

  const float big_neg_f = static_cast<float>(big_neg);
  const int color_yellow = static_cast<int>(std::min<int64_t>(1, n_colors - 1));

  torch::Tensor masked_logits = qty_logits.clone();
  auto bool_opts = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
  torch::Tensor mask = torch::empty({B, Q}, bool_opts);

  float* masked_ptr = masked_logits.data_ptr<float>();
  const float* logits_ptr = qty_logits.data_ptr<float>();
  auto* mask_ptr = mask.data_ptr<bool>();

  std::vector<int64_t> forced_indices(B, 0);
  std::vector<bool> active(B, true);

#pragma omp parallel for if (B > 64)
  for (int64_t b = 0; b < B; ++b) {
    bool* row_mask = mask_ptr + b * Q;
    std::fill(row_mask, row_mask + Q, false);
    RowData row = compute_row_data(
        b,
        cap_reader,
        inv_sum_reader,
        nest_reader,
        inv_reader,
        pick_reader,
        pan_reader,
        pan_room_reader,
        n_colors,
        pick.size(1),
        pan.size(1),
        color_yellow,
        nest_cap,
        nest_blue_id,
        allow_steal,
        can_flip,
        n_verbs);

    int64_t verb_val = verb_reader.get<int64_t>(b);
    int64_t node_val = node_reader.get<int64_t>(b);
    int64_t color_val = color_reader.get<int64_t>(b);

    bool requires_node = (verb_val == kVerbPick) || (verb_val == kVerbPlace) || (verb_val == kVerbSteal);
    bool row_active = verb_val >= 0 && verb_val < n_verbs && color_val >= 0 && color_val < n_colors &&
        (!requires_node || (node_val >= 0 && node_val < n_nodes));

    if (!row_active) {
      active[b] = false;
      row_mask[0] = true;
      continue;
    }

    row_mask[0] = false;
    int32_t limit = 0;
    if (verb_val == kVerbPick) {
      const int64_t pickup_local = pickup_idx_ptr[node_val];
      if (pickup_local >= 0 && pickup_local < pick.size(1) && row.cap > 0) {
        int32_t stock = row.pickup_stock[pickup_local * n_colors + color_val];
        limit = std::min(stock, row.cap);
      }
    } else if (verb_val == kVerbPlace) {
      const int64_t pantry_local = pantry_idx_ptr[node_val];
      if (pantry_local >= 0 && pantry_local < pan.size(1)) {
        int32_t room = row.pantry_room[pantry_local];
        int32_t inv_val = row.inv_colors[color_val];
        limit = std::min(room, inv_val);
      } else if (row.nest_cap_left > 0 && nest_blue_id >= 0 && node_val == nest_blue_id) {
        int32_t inv_val = row.inv_colors[color_val];
        limit = std::min(row.nest_cap_left, inv_val);
      }
    } else if (verb_val == kVerbFlip) {
      int32_t inv_val = row.inv_colors[color_val];
      limit = row.total_inv - inv_val;
    } else if (verb_val == kVerbSteal) {
      const int64_t pantry_local = pantry_idx_ptr[node_val];
      if (pantry_local >= 0 && pantry_local < pan.size(1) && row.cap > 0) {
        int32_t stock = row.pantry_stock[pantry_local * n_colors + color_val];
        limit = std::min(stock, row.cap);
      }
    }

    limit = clamp_positive(limit);
    if (limit > Q - 1) {
      limit = static_cast<int32_t>(Q - 1);
    }
    for (int32_t q = 1; q <= limit; ++q) {
      row_mask[q] = true;
    }

    if (Q > 1) {
      ensure_mask_has_true(row_mask, Q, logits_ptr + b * Q, 1);
    } else {
      ensure_mask_has_true(row_mask, Q, logits_ptr + b * Q, 0);
    }

    float* row_logits = masked_ptr + b * Q;
    row_logits[0] = big_neg_f;
    for (int64_t q = 1; q < Q; ++q) {
      if (!row_mask[q]) {
        row_logits[q] = big_neg_f;
      }
    }
  }

  auto uniform = torch::empty_like(masked_logits).uniform_(1e-6f, 1.0f);
  auto gumbel = -torch::log(-torch::log(uniform));
  auto noisy = masked_logits + gumbel;
  auto max_pair = noisy.max(1);
  torch::Tensor indices = std::get<1>(max_pair);
  torch::Tensor log_soft = torch::log_softmax(masked_logits, 1);
  torch::Tensor logp = log_soft.gather(1, indices.unsqueeze(1)).squeeze(1);
  torch::Tensor prob = torch::softmax(masked_logits, 1);
  torch::Tensor entropy = (-prob * log_soft).sum(1);

  auto* indices_ptr = indices.data_ptr<int64_t>();
  auto* logp_ptr = logp.data_ptr<float>();
  auto* entropy_ptr = entropy.data_ptr<float>();
  for (int64_t b = 0; b < B; ++b) {
    if (!active[b]) {
      indices_ptr[b] = forced_indices[b];
      logp_ptr[b] = 0.0f;
      entropy_ptr[b] = 0.0f;
    }
  }

  return std::make_tuple(indices, logp, entropy);
}

torch::Tensor apply_qty_limits_to_logits(
    const torch::Tensor& qty_logits,
    const torch::Tensor& limit,
    double big_neg) {
  TORCH_CHECK(qty_logits.device().is_cpu(), "qty_logits must be on CPU");
  TORCH_CHECK(limit.device().is_cpu(), "limit must be on CPU");
  TORCH_CHECK(qty_logits.scalar_type() == torch::kFloat32, "qty_logits must be float32");
  TORCH_CHECK(qty_logits.is_contiguous(), "qty_logits must be contiguous");
  TORCH_CHECK(limit.is_contiguous(), "limit must be contiguous");
  TORCH_CHECK(qty_logits.dim() == 2, "qty_logits must be 2D");
  TORCH_CHECK(limit.dim() == 1 && limit.size(0) == qty_logits.size(0), "limit shape mismatch");

  const int64_t B = qty_logits.size(0);
  const int64_t Q = qty_logits.size(1);

  Reader1D limit_reader(limit);

  auto out = qty_logits.clone();
  float* out_ptr = out.data_ptr<float>();
  const float* in_ptr = qty_logits.data_ptr<float>();
  const float big_neg_f = static_cast<float>(big_neg);

#pragma omp parallel for if (B > 64)
  for (int64_t b = 0; b < B; ++b) {
    float* out_row = out_ptr + b * Q;
    const float* in_row = in_ptr + b * Q;
    int32_t lim = clamp_positive(limit_reader.get<int32_t>(b));
    if (lim > Q - 1) {
      lim = static_cast<int32_t>(Q - 1);
    }
    out_row[0] = big_neg_f;
    if (lim >= 1) {
      for (int64_t q = 1; q < Q; ++q) {
        out_row[q] = (q <= lim) ? in_row[q] : big_neg_f;
      }
    } else {
      int64_t best = 1;
      float best_val = (Q > 1) ? in_row[1] : big_neg_f;
      for (int64_t q = 2; q < Q; ++q) {
        if (in_row[q] > best_val) {
          best_val = in_row[q];
          best = q;
        }
        out_row[q] = big_neg_f;
      }
      if (Q > 1) {
        out_row[best] = in_row[best];
      }
    }
  }

  return out;
}

#ifdef BUILD_TESTS
void run_tests() {
  const int64_t B = 3;
  const int64_t n_verbs = 4;
  const int64_t n_nodes = 6;
  const int64_t n_colors = 2;
  const int64_t Q = 5;
  const int64_t n_pantries = 2;
  const int64_t n_pickups = 2;

  auto tensor_opt = torch::TensorOptions().dtype(torch::kFloat32);
  auto verb_logits = torch::tensor(
      {{1.0f, 0.5f, -0.2f, -1.0f},
       {0.0f, 1.0f, 0.0f, 0.0f},
       {-0.5f, -0.1f, 0.3f, 0.2f}},
      tensor_opt);
  auto node_logits = torch::ones({B, n_nodes}, tensor_opt);
  auto color_logits = torch::zeros({B, n_colors}, tensor_opt);
  auto qty_logits = torch::linspace(0, 1, B * Q, tensor_opt).view({B, Q});

  auto blue_node = torch::tensor({0, 1, 2}, torch::TensorOptions().dtype(torch::kLong));
  auto inv_b = torch::tensor(
      {{1, 0},
       {2, 1},
       {0, 3}},
      torch::TensorOptions().dtype(torch::kInt));
  auto inv_sum = torch::tensor({1, 3, 3}, torch::TensorOptions().dtype(torch::kInt));
  auto cap_left = torch::tensor({2, 1, 0}, torch::TensorOptions().dtype(torch::kInt));
  auto nest_blue = torch::tensor({0, 1, 2}, torch::TensorOptions().dtype(torch::kInt));

  auto pan = torch::zeros({B, n_pantries, n_colors}, torch::TensorOptions().dtype(torch::kInt));
  pan.index_put_({0, 0, 0}, 1);
  pan.index_put_({1, 1, 1}, 2);
  auto pick = torch::zeros({B, n_pickups, n_colors}, torch::TensorOptions().dtype(torch::kInt));
  pick.index_put_({0, 0, 0}, 2);
  pick.index_put_({1, 0, 1}, 1);
  pick.index_put_({2, 1, 1}, 3);

  auto pan_room = torch::tensor(
      {{2, 0},
       {1, 3},
       {0, 0}},
      torch::TensorOptions().dtype(torch::kInt));
  auto pan_sum = torch::zeros({B, n_pantries}, torch::TensorOptions().dtype(torch::kInt));
  auto pick_sum = torch::zeros({B, n_pickups}, torch::TensorOptions().dtype(torch::kInt));

  auto pantry_idx = torch::tensor({0, -1, 1, -1, -1, -1}, torch::TensorOptions().dtype(torch::kLong));
  auto pickup_idx = torch::tensor({-1, 0, -1, 1, -1, -1}, torch::TensorOptions().dtype(torch::kLong));
  auto pantry_nodes = torch::tensor({0, 2}, torch::TensorOptions().dtype(torch::kLong));
  auto pickup_nodes = torch::tensor({1, 3}, torch::TensorOptions().dtype(torch::kLong));

  auto result = build_masks_and_apply_logits(
      verb_logits,
      node_logits,
      color_logits,
      qty_logits,
      blue_node,
      inv_b,
      inv_sum,
      cap_left,
      pan,
      pick,
      pan_room,
      pan_sum,
      pick_sum,
      nest_blue,
      pantry_idx,
      pickup_idx,
      pantry_nodes,
      pickup_nodes,
      n_verbs,
      n_nodes,
      n_colors,
      Q,
      5,
      5,
      4,
      4,
      true,
      true,
      -1e9,
      true,
      true,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt);

  TORCH_CHECK(std::get<0>(result).sizes() == torch::IntArrayRef({B, n_verbs}), "verb_mask shape error");
  TORCH_CHECK(std::get<1>(result).sizes() == torch::IntArrayRef({B, n_nodes}), "node_mask shape error");
  TORCH_CHECK(std::get<2>(result).sizes() == torch::IntArrayRef({B, n_colors}), "color_mask shape error");
  TORCH_CHECK(std::get<3>(result).sizes() == torch::IntArrayRef({B, Q}), "qty_mask shape error");

  auto verb_sel = torch::tensor({0, 1, 3}, torch::TensorOptions().dtype(torch::kLong));
  auto node_stage = build_masks_and_apply_logits(
      verb_logits,
      node_logits,
      color_logits,
      qty_logits,
      blue_node,
      inv_b,
      inv_sum,
      cap_left,
      pan,
      pick,
      pan_room,
      pan_sum,
      pick_sum,
      nest_blue,
      pantry_idx,
      pickup_idx,
      pantry_nodes,
      pickup_nodes,
      n_verbs,
      n_nodes,
      n_colors,
      Q,
      5,
      5,
      4,
      4,
      true,
      true,
      -1e9,
      false,
      true,
      verb_sel,
      c10::nullopt,
      c10::nullopt);

  TORCH_CHECK(std::get<1>(node_stage).sizes() == torch::IntArrayRef({B, n_nodes}), "node mask stage shape");

  auto node_sel = torch::tensor({1, 0, 2}, torch::TensorOptions().dtype(torch::kLong));
  auto color_stage = build_masks_and_apply_logits(
      verb_logits,
      node_logits,
      color_logits,
      qty_logits,
      blue_node,
      inv_b,
      inv_sum,
      cap_left,
      pan,
      pick,
      pan_room,
      pan_sum,
      pick_sum,
      nest_blue,
      pantry_idx,
      pickup_idx,
      pantry_nodes,
      pickup_nodes,
      n_verbs,
      n_nodes,
      n_colors,
      Q,
      5,
      5,
      4,
      4,
      true,
      true,
      -1e9,
      false,
      true,
      verb_sel,
      node_sel,
      c10::nullopt);

  TORCH_CHECK(std::get<2>(color_stage).sizes() == torch::IntArrayRef({B, n_colors}), "color mask stage shape");

  auto color_sel = torch::tensor({0, 1, 1}, torch::TensorOptions().dtype(torch::kLong));
  auto qty_stage = build_masks_and_apply_logits(
      verb_logits,
      node_logits,
      color_logits,
      qty_logits,
      blue_node,
      inv_b,
      inv_sum,
      cap_left,
      pan,
      pick,
      pan_room,
      pan_sum,
      pick_sum,
      nest_blue,
      pantry_idx,
      pickup_idx,
      pantry_nodes,
      pickup_nodes,
      n_verbs,
      n_nodes,
      n_colors,
      Q,
      5,
      5,
      4,
      4,
      true,
      true,
      -1e9,
      false,
      true,
      verb_sel,
      node_sel,
      color_sel);

  TORCH_CHECK(std::get<3>(qty_stage).sizes() == torch::IntArrayRef({B, Q}), "qty mask stage shape");

  auto limits = qty_limit_given(
      verb_sel,
      node_sel,
      color_sel,
      inv_b,
      cap_left,
      pan,
      pick,
      pan_room,
      nest_blue,
      pantry_idx,
      pickup_idx,
      4,
      4);

  TORCH_CHECK(limits.size(0) == B, "qty_limit_given shape error");
  auto masked_qty = apply_qty_limits_to_logits(qty_logits, limits, -1e9);
  TORCH_CHECK(masked_qty.sizes() == qty_logits.sizes(), "apply_qty_limits_to_logits shape error");
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "build_masks_and_apply_logits",
      &build_masks_and_apply_logits,
      py::arg("verb_logits"),
      py::arg("node_logits"),
      py::arg("color_logits"),
      py::arg("qty_logits"),
      py::arg("blue_node"),
      py::arg("inv_b"),
      py::arg("inv_sum"),
      py::arg("cap_left"),
      py::arg("pan"),
      py::arg("pick"),
      py::arg("pan_room"),
      py::arg("pan_sum"),
      py::arg("pick_sum"),
      py::arg("nest_blue"),
      py::arg("pantry_idx"),
      py::arg("pickup_idx"),
      py::arg("pantry_nodes"),
      py::arg("pickup_nodes"),
      py::arg("n_verbs"),
      py::arg("n_nodes"),
      py::arg("n_colors"),
      py::arg("Q"),
      py::arg("capacity"),
      py::arg("pantry_cap"),
      py::arg("nest_cap"),
      py::arg("nest_blue_id"),
      py::arg("allow_steal"),
      py::arg("can_flip"),
      py::arg("big_neg"),
      py::arg("apply_to_logits") = true,
      py::arg("flip_node_all") = true,
      py::arg("verb_sel") = py::none(),
      py::arg("node_sel") = py::none(),
      py::arg("color_sel") = py::none());

  m.def(
      "sample_verb",
      &sample_verb,
      py::arg("verb_logits"),
      py::arg("blue_node"),
      py::arg("inv_b"),
      py::arg("inv_sum"),
      py::arg("cap_left"),
      py::arg("pan"),
      py::arg("pick"),
      py::arg("pan_room"),
      py::arg("pan_sum"),
      py::arg("pick_sum"),
      py::arg("nest_blue"),
      py::arg("pantry_idx"),
      py::arg("pickup_idx"),
      py::arg("pantry_nodes"),
      py::arg("pickup_nodes"),
      py::arg("n_verbs"),
      py::arg("n_nodes"),
      py::arg("n_colors"),
      py::arg("Q"),
      py::arg("capacity"),
      py::arg("pantry_cap"),
      py::arg("nest_cap"),
      py::arg("nest_blue_id"),
      py::arg("allow_steal"),
      py::arg("can_flip"),
      py::arg("big_neg"),
      py::arg("flip_node_all"));

  m.def(
      "sample_node",
      &sample_node,
      py::arg("node_logits"),
      py::arg("blue_node"),
      py::arg("inv_b"),
      py::arg("inv_sum"),
      py::arg("cap_left"),
      py::arg("pan"),
      py::arg("pick"),
      py::arg("pan_room"),
      py::arg("pan_sum"),
      py::arg("pick_sum"),
      py::arg("nest_blue"),
      py::arg("pantry_idx"),
      py::arg("pickup_idx"),
      py::arg("pantry_nodes"),
      py::arg("pickup_nodes"),
      py::arg("verb_idx"),
      py::arg("n_verbs"),
      py::arg("n_nodes"),
      py::arg("n_colors"),
      py::arg("Q"),
      py::arg("capacity"),
      py::arg("pantry_cap"),
      py::arg("nest_cap"),
      py::arg("nest_blue_id"),
      py::arg("allow_steal"),
      py::arg("can_flip"),
      py::arg("big_neg"),
      py::arg("flip_node_all"));

  m.def(
      "sample_color",
      &sample_color,
      py::arg("color_logits"),
      py::arg("blue_node"),
      py::arg("inv_b"),
      py::arg("inv_sum"),
      py::arg("cap_left"),
      py::arg("pan"),
      py::arg("pick"),
      py::arg("pan_room"),
      py::arg("pan_sum"),
      py::arg("pick_sum"),
      py::arg("nest_blue"),
      py::arg("pantry_idx"),
      py::arg("pickup_idx"),
      py::arg("pantry_nodes"),
      py::arg("pickup_nodes"),
      py::arg("verb_idx"),
      py::arg("node_idx"),
      py::arg("n_verbs"),
      py::arg("n_nodes"),
      py::arg("n_colors"),
      py::arg("Q"),
      py::arg("capacity"),
      py::arg("pantry_cap"),
      py::arg("nest_cap"),
      py::arg("nest_blue_id"),
      py::arg("allow_steal"),
      py::arg("can_flip"),
      py::arg("big_neg"),
      py::arg("flip_node_all"));

  m.def(
      "sample_qty",
      &sample_qty,
      py::arg("qty_logits"),
      py::arg("blue_node"),
      py::arg("inv_b"),
      py::arg("inv_sum"),
      py::arg("cap_left"),
      py::arg("pan"),
      py::arg("pick"),
      py::arg("pan_room"),
      py::arg("pan_sum"),
      py::arg("pick_sum"),
      py::arg("nest_blue"),
      py::arg("pantry_idx"),
      py::arg("pickup_idx"),
      py::arg("pantry_nodes"),
      py::arg("pickup_nodes"),
      py::arg("verb_idx"),
      py::arg("node_idx"),
      py::arg("color_idx"),
      py::arg("n_verbs"),
      py::arg("n_nodes"),
      py::arg("n_colors"),
      py::arg("Q"),
      py::arg("capacity"),
      py::arg("pantry_cap"),
      py::arg("nest_cap"),
      py::arg("nest_blue_id"),
      py::arg("allow_steal"),
      py::arg("can_flip"),
      py::arg("big_neg"),
      py::arg("flip_node_all"));

  m.def(
      "qty_limit_given",
      &qty_limit_given,
      py::arg("verb_sel"),
      py::arg("node_sel"),
      py::arg("color_sel"),
      py::arg("inv_b"),
      py::arg("cap_left"),
      py::arg("pan"),
      py::arg("pick"),
      py::arg("pan_room"),
      py::arg("nest_blue"),
      py::arg("pantry_idx"),
      py::arg("pickup_idx"),
      py::arg("nest_cap"),
      py::arg("nest_blue_id"));

  m.def(
      "apply_qty_limits_to_logits",
      &apply_qty_limits_to_logits,
      py::arg("qty_logits"),
      py::arg("limit"),
      py::arg("big_neg"));

#ifdef BUILD_TESTS
  m.def("run_tests", &run_tests);
#endif
}

/*
Example usage:
from torch.utils.cpp_extension import load
ext = load(
    name="mask_ext",
    sources=["mask_ext.cpp"],
    extra_cflags=["-O3", "-fopenmp"],
    extra_ldflags=["-fopenmp"],
    verbose=False,
)

# First stage: mask verbs
verb_mask, node_mask, color_mask, qty_mask, v_log, n_log, c_log, q_log = ext.build_masks_and_apply_logits(
    verb_logits, node_logits, color_logits, qty_logits,
    blue_node, inv_b, inv_sum, cap_left,
    pan, pick, pan_room, pan_sum, pick_sum, nest_blue,
    pantry_idx, pickup_idx, pantry_nodes, pickup_nodes,
    n_verbs, n_nodes, n_colors, Q,
    capacity, pantry_cap, nest_cap, nest_blue_id,
    allow_steal, can_flip,
    big_neg, True, True)

# Subsequent stages: provide selected verb/node/color
node_stage = ext.build_masks_and_apply_logits(
    verb_logits, node_logits, color_logits, qty_logits,
    blue_node, inv_b, inv_sum, cap_left,
    pan, pick, pan_room, pan_sum, pick_sum, nest_blue,
    pantry_idx, pickup_idx, pantry_nodes, pickup_nodes,
    n_verbs, n_nodes, n_colors, Q,
    capacity, pantry_cap, nest_cap, nest_blue_id,
    allow_steal, can_flip,
    big_neg, True, True,
    verb_sel)

color_stage = ext.build_masks_and_apply_logits(
    verb_logits, node_logits, color_logits, qty_logits,
    blue_node, inv_b, inv_sum, cap_left,
    pan, pick, pan_room, pan_sum, pick_sum, nest_blue,
    pantry_idx, pickup_idx, pantry_nodes, pickup_nodes,
    n_verbs, n_nodes, n_colors, Q,
    capacity, pantry_cap, nest_cap, nest_blue_id,
    allow_steal, can_flip,
    big_neg, True, True,
    verb_sel, node_sel)

qty_stage = ext.build_masks_and_apply_logits(
    verb_logits, node_logits, color_logits, qty_logits,
    blue_node, inv_b, inv_sum, cap_left,
    pan, pick, pan_room, pan_sum, pick_sum, nest_blue,
    pantry_idx, pickup_idx, pantry_nodes, pickup_nodes,
    n_verbs, n_nodes, n_colors, Q,
    capacity, pantry_cap, nest_cap, nest_blue_id,
    allow_steal, can_flip,
    big_neg, True, True,
    verb_sel, node_sel, color_sel)
*/
