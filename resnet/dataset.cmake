################################## Handle the dataset and extract them.  ###############################################
option(GENERATE_DATASET "Generate dataset" ON)

# Check if have correct Python

if (NOT GENERATE_DATASET)
    message(STATUS "Dataset generation is disabled")
    message(STATUS "Set CMake option GENERATE_DATASET to ON to enable it")
    return()
endif()

execute_process(COMMAND
        python "${PROJECT_SOURCE_DIR}/src/python_helpers/test_dependencies.py"
        RESULT_VARIABLE PYTHON_ERROR)

if (PYTHON_ERROR)
    message(WARNING "Python does not have the correct dependencies. Skipping.")
    return()
endif ()

# Config the dataset path.
set(DATASET_PATH "${PROJECT_SOURCE_DIR}/../dataset" CACHE PATH "The path of the dataset.")
message(STATUS "Default dataset path: ${DATASET_PATH}")
message(STATUS "Change it by setting the CMake DATASET_PATH variable.")

message(CHECK_START "Check if dataset is present")
# if dataset exists, then extract it.
if (EXISTS "${DATASET_PATH}/imagenet.tar.gz")
    message(CHECK_PASS "found")
else ()
    message(CHECK_FAIL "Not found")
    message(WARNING "Dataset not found. Check README.md for instructions.")
    return()
endif ()

set(PROJECT_HAS_DATASET ON)
set(PROJECT_DATASET_PATH "${PROJECT_BINARY_DIR}/imagenet")
set(PROJECT_DATASET_TENSOR_PATH "${PROJECT_BINARY_DIR}/imagenet_tensor")
set(PROJECT_NETWORK_WEIGHT_PATH "${PROJECT_BINARY_DIR}/weight")
set(PROJECT_TEST_DATA_PATH "${PROJECT_BINARY_DIR}/test_data")

# Make the directory for the dataset.
file(MAKE_DIRECTORY "${PROJECT_DATASET_PATH}")
file(MAKE_DIRECTORY "${PROJECT_DATASET_TENSOR_PATH}")
file(MAKE_DIRECTORY "${PROJECT_NETWORK_WEIGHT_PATH}")
file(MAKE_DIRECTORY "${PROJECT_TEST_DATA_PATH}")

message(CHECK_START "Extracting the dataset")
# Extract the raw dataset.
execute_process(COMMAND
        tar xf ${DATASET_PATH}/imagenet.tar.gz --strip-components=1
        WORKING_DIRECTORY "${PROJECT_DATASET_PATH}"
        RESULT_VARIABLE TAR_ERROR)
if (TAR_ERROR)
    message(CHECK_FAIL "Failed")
    message(WARNING "Failed to extract the dataset: ${TAR_ERROR}")

    unset(PROJECT_HAS_DATASET)
    unset(PROJECT_DATASET_PATH)
    unset(PROJECT_DATASET_TENSOR_PATH)
    unset(PROJECT_NETWORK_WEIGHT_PATH)
    unset(PROJECT_TEST_DATA_PATH)
    return()
endif ()

message(CHECK_PASS "Done")

message(CHECK_START "Preprocessing the dataset and weight.")
# Invoke preprocess.py
execute_process(COMMAND
        python "${PROJECT_SOURCE_DIR}/src/python_helpers/preprocess.py"
        "--dataset-dir=${PROJECT_DATASET_PATH}"
        "--tensor-output-dir=${PROJECT_DATASET_TENSOR_PATH}"
        "--network-output-dir=${PROJECT_NETWORK_WEIGHT_PATH}"
        RESULT_VARIABLE PREPROCESS_ERROR)
if (PREPROCESS_ERROR)
    message(CHECK_FAIL "Failed")
    message(WARNING "Failed to preprocess the dataset")
    unset(PROJECT_HAS_DATASET)
    unset(PROJECT_DATASET_PATH)
    unset(PROJECT_DATASET_TENSOR_PATH)
    unset(PROJECT_NETWORK_WEIGHT_PATH)
    unset(PROJECT_TEST_DATA_PATH)
    return()
else ()
    message(CHECK_PASS "Done")
endif ()

if (PROJECT_HAS_DATASET)
    message(STATUS "Generated dataset tensor path: ${PROJECT_DATASET_PATH}")
    set(DATASET_DEFINITIONS "${DATASET_DEFINITIONS};WITH_DATA")
    set(DATASET_DEFINITIONS "${DATASET_DEFINITIONS};DATASET_ROOT=\"${PROJECT_DATASET_TENSOR_PATH}\"")
    set(DATASET_DEFINITIONS "${DATASET_DEFINITIONS};RESNET18_ROOT=\"${PROJECT_NETWORK_WEIGHT_PATH}\"")
    set(DATASET_DEFINITIONS "${DATASET_DEFINITIONS};TEST_DATA_ROOT=\"${PROJECT_TEST_DATA_PATH}\"")
endif ()


