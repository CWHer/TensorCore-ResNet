################################## Handle the dataset and extract them.  ###############################################

# Config the dataset path.
set(DATASET_PATH "${PROJECT_SOURCE_DIR}/../dataset" CACHE PATH "The path of the dataset.")
message(STATUS "Dataset Path: ${DATASET_PATH}")

# if dataset exists, then extract it.
if (EXISTS "${DATASET_PATH}/imagenet.tar")
    message(STATUS "Dataset Exists.")
    file(SHA1 "${DATASET_PATH}/imagenet.tar" imagenet.tar.sha1sum)
    if (NOT imagenet.tar.sha1sum STREQUAL "1a4f979033b3609b6a5a502ba9a2384ea957360c")
        message(FATAL_ERROR "The sha1sum of imagenet.tar is not correct.")

    else ()
        set(PROJECT_HAS_DATASET ON)
        set(PROJECT_DATASET_PATH "${PROJECT_BINARY_DIR}/imagenet")
        set(PROJECT_DATASET_TENSOR_PATH "${PROJECT_BINARY_DIR}/imagenet_tensor")
        # Make the directory for the dataset.
        file(MAKE_DIRECTORY "${PROJECT_DATASET_PATH}")
        execute_process(COMMAND
                tar xf ${DATASET_PATH}/imagenet.tar --strip-components=6
                WORKING_DIRECTORY "${PROJECT_DATASET_PATH}"
                RESULT_VARIABLE TAR_ERROR)
        if (TAR_ERROR)
            message(WARNING "Failed to extract the dataset: ${TAR_ERROR}")
            unset(PROJECT_HAS_DATASET)
            unset(PROJECT_DATASET_PATH)
            unset(PROJECT_DATASET_TENSOR_PATH)
        else ()
            message(STATUS "Dataset Extracted.")
            # Invoke preprocess.py
            execute_process(COMMAND
                    python "${PROJECT_SOURCE_DIR}/preprocess.py" "--dataset-dir=${PROJECT_DATASET_PATH}"
                    "--output-dir=${PROJECT_DATASET_TENSOR_PATH}"
                    RESULT_VARIABLE PREPROCESS_ERROR)
            if (PREPROCESS_ERROR)
                message(WARNING "Failed to preprocess the dataset: ${PREPROCESS_ERROR}")
                unset(PROJECT_HAS_DATASET)
                unset(PROJECT_DATASET_PATH)
                unset(PROJECT_DATASET_TENSOR_PATH)
            else ()
                message(STATUS "Dataset Preprocessed.")
            endif ()
        endif ()
    endif ()
else ()
    message(STATUS "Dataset not exist.")
endif ()

if (PROJECT_HAS_DATASET)
    message(STATUS "Generated dataset tensor Path: ${PROJECT_DATASET_PATH}")
    set(TEST_ADDITIONAL_DEFINITIONS "${TEST_ADDITIONAL_DEFINITIONS};WITH_DATASET")
    set(TEST_ADDITIONAL_DEFINITIONS "${TEST_ADDITIONAL_DEFINITIONS};DATASET_ROOT=\"${PROJECT_DATASET_TENSOR_PATH}\"")
endif ()

# Add the parameters
if (EXISTS "${DATASET_PATH}/resnet18.tar")
    message(STATUS "ResNet18 parameters exists.")

    file(SHA1 "${DATASET_PATH}/resnet18.tar" resnet18.tar.sha1sum)
    if (NOT resnet18.tar.sha1sum STREQUAL "cda58c4d0486a94578bc614652e64f471119657a")
        message(FATAL_ERROR "The sha1sum of resnet18.tar is not correct.")
    else ()
        set(PROJECT_HAS_RESNET18 ON)
        set(PROJECT_RESNET18_PATH "${PROJECT_BINARY_DIR}/resnet18")
        file(MAKE_DIRECTORY "${PROJECT_RESNET18_PATH}")
        execute_process(COMMAND
                tar xf ${DATASET_PATH}/resnet18.tar --strip-components=1
                WORKING_DIRECTORY "${PROJECT_RESNET18_PATH}"
                RESULT_VARIABLE TAR_ERROR)
        if (TAR_ERROR)
            message(WARNING "Failed to extract the resnet18 parameters: ${TAR_ERROR}")
            unset(PROJECT_HAS_RESNET18)
            unset(PROJECT_RESNET18_PATH)
        else ()
            message(STATUS "ResNet18 parameters extracted.")
        endif ()
    endif ()
endif ()

if (PROJECT_HAS_RESNET18)
    message(STATUS "Generated ResNet18 parameters Path: ${PROJECT_RESNET18_PATH}")

    set(TEST_ADDITIONAL_DEFINITIONS "${TEST_ADDITIONAL_DEFINITIONS};WITH_RESNET18")
    set(TEST_ADDITIONAL_DEFINITIONS "${TEST_ADDITIONAL_DEFINITIONS};RESNET18_ROOT=\"${PROJECT_RESNET18_PATH}\"")
endif ()

