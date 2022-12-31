# ################################# Handle the dataset and extract them.  ###############################################
# Check if have correct Python

execute_process(COMMAND
    python "${PROJECT_SOURCE_DIR}/py_utils/test_dependencies.py"
    RESULT_VARIABLE PYTHON_ERROR)

if(PYTHON_ERROR)
    message(WARNING "Python does not have the correct dependencies. Skipping.")
    return()
endif()

# Config the dataset path.
set(DATASET_PATH "${PROJECT_SOURCE_DIR}/../dataset" CACHE PATH "The path of the dataset.")
message(STATUS "Dataset Path: ${DATASET_PATH}")
message(STATUS "Change it by setting the DATASET_PATH variable.")

# TODO: imagenet.tar is a demo tar file with only 100 images.
# if dataset exists, then extract it.
if(EXISTS "${DATASET_PATH}/imagenet.tar")
    message(STATUS "Dataset Exists.")
    file(SHA1 "${DATASET_PATH}/imagenet.tar" imagenet.tar.sha1sum)

    if(NOT imagenet.tar.sha1sum STREQUAL "1a4f979033b3609b6a5a502ba9a2384ea957360c")
        message(FATAL_ERROR "The sha1sum of imagenet.tar is not correct.")
        return()
    endif()
else()
    message(WARNING "Dataset does not exist. skipped.")
    return()
endif()

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

# Extract the raw dataset.
execute_process(COMMAND
    tar xf ${DATASET_PATH}/imagenet.tar --strip-components=6
    WORKING_DIRECTORY "${PROJECT_DATASET_PATH}"
    RESULT_VARIABLE TAR_ERROR)

if(TAR_ERROR)
    message(WARNING "Failed to extract the dataset: ${TAR_ERROR}")

    unset(PROJECT_HAS_DATASET)
    unset(PROJECT_DATASET_PATH)
    unset(PROJECT_DATASET_TENSOR_PATH)
    unset(PROJECT_NETWORK_WEIGHT_PATH)
    unset(PROJECT_TEST_DATA_PATH)
    return()
endif()

message(STATUS "Dataset Extracted.")

# Invoke preprocess.py
execute_process(COMMAND
    python "${PROJECT_SOURCE_DIR}/py_utils/preprocess.py"
    "--dataset-dir=${PROJECT_DATASET_PATH}"
    "--dataset-output-dir=${PROJECT_DATASET_TENSOR_PATH}"
    "--network-output-dir=${PROJECT_NETWORK_WEIGHT_PATH}"
    RESULT_VARIABLE PREPROCESS_ERROR)

if(PREPROCESS_ERROR)
    message(WARNING "Failed to preprocess the dataset")
    unset(PROJECT_HAS_DATASET)
    unset(PROJECT_DATASET_PATH)
    unset(PROJECT_DATASET_TENSOR_PATH)
    unset(PROJECT_NETWORK_WEIGHT_PATH)
    unset(PROJECT_TEST_DATA_PATH)
else()
    message(STATUS "Dataset Preprocessed.")
endif()

if(PROJECT_HAS_DATASET)
    message(STATUS "Generated dataset tensor Path: ${PROJECT_DATASET_PATH}")
    set(MAIN_ADDITIONAL_DEFINITIONS "${MAIN_ADDITIONAL_DEFINITIONS};DATASET_ROOT=\"${PROJECT_DATASET_TENSOR_PATH}\"")
    set(MAIN_ADDITIONAL_DEFINITIONS "${MAIN_ADDITIONAL_DEFINITIONS};RESNET18_ROOT=\"${PROJECT_NETWORK_WEIGHT_PATH}\"")
    set(TEST_ADDITIONAL_DEFINITIONS "${TEST_ADDITIONAL_DEFINITIONS};DATASET_ROOT=\"${PROJECT_DATASET_TENSOR_PATH}\"")
    set(TEST_ADDITIONAL_DEFINITIONS "${TEST_ADDITIONAL_DEFINITIONS};RESNET18_ROOT=\"${PROJECT_NETWORK_WEIGHT_PATH}\"")
    set(TEST_ADDITIONAL_DEFINITIONS "${TEST_ADDITIONAL_DEFINITIONS};TEST_DATA_ROOT=\"${PROJECT_TEST_DATA_PATH}\"")
endif()
