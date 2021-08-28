/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <CL/cl.h>

#include "oneapi/dnnl/dnnl_ocl.h"

#include "common/c_types_map.hpp"
#include "common/engine.hpp"
#include "common/memory.hpp"
#include "common/utils.hpp"

#include "gpu/ocl/ocl_c_types_map.hpp"
#include "gpu/ocl/ocl_engine.hpp"
#include "gpu/ocl/ocl_memory_storage.hpp"
#include "gpu/ocl/ocl_stream.hpp"

using namespace dnnl::impl;

status_t dnnl_ocl_interop_primitive_execute(
        const primitive_iface_t *primitive_iface, stream_t *stream, int nargs,
        const dnnl_exec_arg_t *args, const cl_event *deps, int ndeps,
        cl_event *return_event_) {
    bool ok = !utils::any_null(primitive_iface, stream)
            && primitive_iface->engine() == stream->engine()
            && primitive_iface->engine()->runtime_kind() == runtime_kind::ocl
            && IMPLICATION(nargs > 0, args != nullptr);
    if (!ok) return status::invalid_arguments;

    auto *ocl_stream = utils::downcast<gpu::ocl::ocl_stream_t *>(stream);

    if (ocl_stream->flags() & stream_flags::in_order) {
        return status::invalid_arguments;
    }

    if (deps != nullptr) {
        std::vector<cl_event> events(ndeps);
        for (int i = 0; i < ndeps; i++) {
            events[i] = deps[i];
        }
        ocl_stream->set_deps(events);
    }

    // run primitive
    exec_args_t exec_args;
    CHECK(cvt_primitive_args(
            primitive_iface->pd()->impl().get(), nargs, args, exec_args));

    exec_ctx_t ctx(ocl_stream, std::move(exec_args));
    CHECK(primitive_execute(primitive_iface, ctx));

    // return output event
    auto last_events = ocl_stream->get_deps();
    if (last_events.size() != 1) {
        assert(!"unexpected");
        return status::runtime_error;
    }

    cl_event return_event = last_events[0];

    if (return_event_ != nullptr)
        *return_event_ = return_event;
    else
        clReleaseEvent(return_event);

    return status::success;
}
