#include <xnnpack.h>
#include <iostream>
#include <memory>
#include <vector>
#include <cassert>
#include <limits>
#include <ctime>

#include <chrono>

void create_and_pack_weights(xnn_weights_cache_t& cache, bool finalized){
    size_t input_channels = 20000;
    size_t output_channels = 40000;
    xnn_subgraph_t subgraph_ptr = nullptr;
    xnn_status status = xnn_create_subgraph(
        /*external_value_ids=*/2,
        /*flags=*/0,
        &subgraph_ptr);

    assert(status == xnn_status_success);
    std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> subgraph(
        subgraph_ptr, &xnn_delete_subgraph);


    uint32_t input_id;
    std::vector<size_t> input_dims{4, 4, 20, input_channels};
    // Define input
    status = xnn_define_tensor_value(
        subgraph_ptr,
        xnn_datatype_fp32,
        input_dims.size(),
        input_dims.data(),
        nullptr,
        0,
        XNN_VALUE_FLAG_EXTERNAL_INPUT,
        &input_id);
    assert(status == xnn_status_success);

    // define weight
    uint32_t weight_id;
    std::vector<float> weight_data(output_channels * input_channels, 0.0f);
    std::vector<size_t> weight_dims{output_channels, input_channels};
    status = xnn_define_tensor_value(
        subgraph_ptr,
        xnn_datatype_fp32,
        weight_dims.size(),
        weight_dims.data(),
        weight_data.data(),
        XNN_INVALID_VALUE_ID,
        0,
        &weight_id);
    assert(status == xnn_status_success);


    // define bias
    uint32_t bias_id;
    std::vector<float> bias_data(output_channels, 0.0f);
    std::vector<size_t> bias_dims{output_channels};
    status = xnn_define_tensor_value(
        subgraph_ptr,
        xnn_datatype_fp32,
        bias_dims.size(),
        bias_dims.data(),
        bias_data.data(),
        XNN_INVALID_VALUE_ID,
        0,
        &bias_id);
    assert(status == xnn_status_success);


    // define output
    uint32_t output_id;
    std::vector<size_t> output_dims{4, 4, 20, output_channels};
    status = xnn_define_tensor_value(
        subgraph_ptr,
        xnn_datatype_fp32,
        output_dims.size(),
        output_dims.data(),
        nullptr,
        1,
        XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
        &output_id);
    assert(status == xnn_status_success);

    // create fully connected
    status = xnn_define_fully_connected(
        subgraph_ptr,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        input_id,
        weight_id,
        bias_id,
        output_id,
        0
    );

    xnn_runtime_t runtime_ptr = nullptr;
    std::cout << "megabytes Packed: " << (input_channels * output_channels * 4)/1000000 << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    status = xnn_create_runtime_v3(
        subgraph.get(),
        cache,
        nullptr,
        0,
        &runtime_ptr);
    assert(status == xnn_status_success);
    auto end = std::chrono::high_resolution_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    std::cout << "pack time: " << ms_int.count() << std::endl;


    if (!finalized) {
        status = xnn_finalize_weights_cache(cache, xnn_weights_cache_finalization_kind_soft);
        assert(status == xnn_status_success);
    }

    // std::vector<float> input_data;
    // input_data.resize(4 * 4 * 20 * input_channels);
    // std::vector<float> output_data;
    // output_data.resize(4 * 4 * 20 * output_channels);

    // auto start1 = std::chrono::high_resolution_clock::now();
    // status = xnn_reshape_external_value(runtime_ptr, input_id, input_dims.size(), input_dims.data());
    // assert(status == xnn_status_success);
    // status = xnn_reshape_runtime(runtime_ptr);
    // assert(status == xnn_status_success);

    // std::vector<xnn_external_value> external_values = {
    //     xnn_external_value{input_id, input_data.data()},
    //     xnn_external_value{output_id, output_data.data()},

    // };

    // status = xnn_setup_runtime_v2(runtime_ptr, external_values.size(), external_values.data());
    // assert(status == xnn_status_success);

    // status = xnn_invoke_runtime(runtime_ptr);
    // assert(status == xnn_status_success);
    // auto end1 = std::chrono::high_resolution_clock::now();

    // auto ms_int1 = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - start1);
    // std::cout << "invoke time " << ms_int1.count() << std::endl;

}

int main() {
    xnn_initialize(nullptr);
    xnn_weights_cache_t cache = nullptr;
    xnn_status status = xnn_create_weights_cache_with_size(4000000000, &cache);
    assert(status == xnn_status_success);

    create_and_pack_weights(cache, false);
    std::cout << "using weights cache now" << std::endl;
    create_and_pack_weights(cache, true);

    return 0;
}
