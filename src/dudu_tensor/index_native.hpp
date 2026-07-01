#pragma once

#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ddtensor {

struct IndexPlan {
    std::vector<int> shape{};
    std::vector<int> strides{};
    int offset{};
};

enum class IndexKind { Scalar, Slice, Ellipsis, NewAxis };

struct IndexItem {
    IndexKind kind{};
    int scalar{};
    bool has_start{};
    bool has_end{};
    bool has_step{};
    int start{};
    int end{};
    int step{1};
};

inline int normalize_axis_index(int value, int extent) {
    int out = value < 0 ? value + extent : value;
    if (out < 0 || out >= extent) {
        throw std::out_of_range("tensor index out of range");
    }
    return out;
}

inline int normalized_slice_bound(int value, int extent) {
    int out = value < 0 ? value + extent : value;
    if (out < 0) {
        return 0;
    }
    if (out > extent) {
        return extent;
    }
    return out;
}

inline int slice_count(int start, int stop, int step) {
    if (step <= 0) {
        throw std::invalid_argument("tensor slices require positive step");
    }
    if (stop <= start) {
        return 0;
    }
    return ((stop - start) + step - 1) / step;
}

inline int element_count(const std::vector<int>& shape) {
    int total = 1;
    for (int dim : shape) {
        total *= dim;
    }
    return shape.empty() ? 1 : total;
}

inline int flat_offset(const std::vector<int>& shape, const std::vector<int>& strides, int offset,
                       int flat) {
    int source_offset = offset;
    for (std::size_t dim = shape.size(); dim > 0; --dim) {
        const std::size_t axis = dim - 1;
        const int extent = shape[axis];
        const int coord = extent == 0 ? 0 : flat % extent;
        flat = extent == 0 ? 0 : flat / extent;
        source_offset += coord * strides[axis];
    }
    return source_offset;
}

template <class T>
concept IntegralIndex = std::is_integral_v<std::decay_t<T>>;

template <class T>
concept SliceLike = requires(const T& value) {
    value.has_start;
    value.has_end;
    value.has_step;
    value.start;
    value.end;
    value.step;
};

template <class T>
concept ScalarIndexLike = requires(const T& value) {
    value.value;
};

template <class T>
concept BasicIndexLike = requires(const T& value) {
    value.kind;
    value.scalar;
    value.slice;
};

template <IntegralIndex T> IndexItem index_item(T value) {
    return {.kind = IndexKind::Scalar, .scalar = static_cast<int>(value)};
}

template <ScalarIndexLike T> IndexItem index_item(const T& value) {
    return {.kind = IndexKind::Scalar, .scalar = static_cast<int>(value.value)};
}

template <SliceLike T> IndexItem index_item(const T& value) {
    return {
        .kind = IndexKind::Slice,
        .has_start = value.has_start,
        .has_end = value.has_end,
        .has_step = value.has_step,
        .start = static_cast<int>(value.start),
        .end = static_cast<int>(value.end),
        .step = static_cast<int>(value.step),
    };
}

template <BasicIndexLike T> IndexItem index_item(const T& value) {
    const int kind = static_cast<int>(value.kind);
    if (kind == 0) {
        return {.kind = IndexKind::Scalar, .scalar = static_cast<int>(value.scalar)};
    }
    if (kind == 1) {
        return index_item(value.slice);
    }
    if (kind == 2) {
        return {.kind = IndexKind::Ellipsis};
    }
    if (kind == 3) {
        return {.kind = IndexKind::NewAxis};
    }
    throw std::invalid_argument("unknown basic index kind");
}

inline int consumed_axes(const std::vector<IndexItem>& items) {
    int count = 0;
    for (const IndexItem& item : items) {
        if (item.kind == IndexKind::Scalar || item.kind == IndexKind::Slice) {
            count += 1;
        }
    }
    return count;
}

inline std::vector<IndexItem> expand_ellipsis(const std::vector<IndexItem>& items,
                                              int source_rank) {
    std::vector<IndexItem> out;
    bool saw_ellipsis = false;
    const int consumed = consumed_axes(items);
    for (const IndexItem& item : items) {
        if (item.kind != IndexKind::Ellipsis) {
            out.push_back(item);
            continue;
        }
        if (saw_ellipsis) {
            throw std::invalid_argument("index expression may contain at most one ellipsis");
        }
        saw_ellipsis = true;
        const int fill = source_rank - consumed;
        if (fill < 0) {
            throw std::invalid_argument("too many indices for tensor");
        }
        for (int i = 0; i < fill; ++i) {
            out.push_back({.kind = IndexKind::Slice});
        }
    }
    return out;
}

inline IndexPlan index_plan_from_items(const std::vector<int>& source_shape,
                                       const std::vector<int>& source_strides, int source_offset,
                                       std::vector<IndexItem> items) {
    if (source_shape.size() != source_strides.size()) {
        throw std::invalid_argument("tensor shape/stride rank mismatch");
    }
    items = expand_ellipsis(items, static_cast<int>(source_shape.size()));

    IndexPlan plan;
    plan.offset = source_offset;
    int axis = 0;
    for (const IndexItem& item : items) {
        if (item.kind == IndexKind::NewAxis) {
            plan.shape.push_back(1);
            plan.strides.push_back(0);
            continue;
        }
        if (axis >= static_cast<int>(source_shape.size())) {
            throw std::invalid_argument("too many indices for tensor");
        }
        const int extent = source_shape[static_cast<std::size_t>(axis)];
        const int stride = source_strides[static_cast<std::size_t>(axis)];
        if (item.kind == IndexKind::Scalar) {
            plan.offset += normalize_axis_index(item.scalar, extent) * stride;
            axis += 1;
            continue;
        }
        if (item.kind == IndexKind::Slice) {
            const int step = item.has_step ? item.step : 1;
            const int start = item.has_start ? normalized_slice_bound(item.start, extent) : 0;
            const int stop = item.has_end ? normalized_slice_bound(item.end, extent) : extent;
            plan.offset += start * stride;
            plan.shape.push_back(slice_count(start, stop, step));
            plan.strides.push_back(stride * step);
            axis += 1;
            continue;
        }
    }
    while (axis < static_cast<int>(source_shape.size())) {
        plan.shape.push_back(source_shape[static_cast<std::size_t>(axis)]);
        plan.strides.push_back(source_strides[static_cast<std::size_t>(axis)]);
        axis += 1;
    }
    return plan;
}

template <class... Args>
IndexPlan index_plan(const std::vector<int>& source_shape, const std::vector<int>& source_strides,
                     int source_offset, const Args&... args) {
    return index_plan_from_items(source_shape, source_strides, source_offset,
                                 std::vector<IndexItem>{index_item(args)...});
}

template <class... Args>
int element_offset(const std::vector<int>& source_shape, const std::vector<int>& source_strides,
                   int source_offset, const Args&... args) {
    IndexPlan plan = index_plan(source_shape, source_strides, source_offset, args...);
    if (!plan.shape.empty()) {
        throw std::invalid_argument("tensor element access requires scalar indexes");
    }
    return plan.offset;
}

template <class Data>
Data materialize_view(const Data& source, const std::vector<int>& shape,
                      const std::vector<int>& strides, int offset) {
    Data out;
    const int count = element_count(shape);
    out.reserve(static_cast<std::size_t>(count));
    for (int flat = 0; flat < count; ++flat) {
        out.push_back(source[static_cast<std::size_t>(flat_offset(shape, strides, offset, flat))]);
    }
    return out;
}

template <class Data, class Value>
void fill_view(Data& source, const std::vector<int>& shape, const std::vector<int>& strides,
               int offset, const Value& value) {
    const int count = element_count(shape);
    for (int flat = 0; flat < count; ++flat) {
        source[static_cast<std::size_t>(flat_offset(shape, strides, offset, flat))] = value;
    }
}

} // namespace ddtensor
