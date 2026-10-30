# Appended to the generated TFLite CMakeLists by mayhem/build.sh. MAYHEM_BUILD selects the targets:
#   fuzz: libFuzzer harnesses + *-standalone reproducers (sanitized);  kat: the known-answer probe.
set(MAYHEM_DIR "${TF_ROOT}/mayhem")
set(MAYHEM_OUT "${TF_ROOT}" CACHE PATH "where binaries land")

if(MAYHEM_BUILD STREQUAL "fuzz")
  add_library(mayhem_lsan_off OBJECT ${MAYHEM_LSAN_OFF_SRC})
  foreach(h tflite_interp tflite_resize)
    add_executable(${h} ${MAYHEM_DIR}/${h}.cc $<TARGET_OBJECTS:mayhem_lsan_off>)
    target_include_directories(${h} PRIVATE ${MAYHEM_DIR})
    target_link_libraries(${h} PRIVATE tensorflow-lite ${MAYHEM_FUZZ_ENGINE})
    set_target_properties(${h} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${MAYHEM_OUT})

    add_executable(${h}-standalone ${MAYHEM_DIR}/${h}.cc $<TARGET_OBJECTS:mayhem_lsan_off>
                   ${MAYHEM_STANDALONE_MAIN_OBJ})
    target_include_directories(${h}-standalone PRIVATE ${MAYHEM_DIR})
    target_link_libraries(${h}-standalone PRIVATE tensorflow-lite)
    set_target_properties(${h}-standalone PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${MAYHEM_OUT})
  endforeach()
elseif(MAYHEM_BUILD STREQUAL "kat")
  add_executable(tflite_kat ${MAYHEM_DIR}/tflite_kat.cc)
  target_link_libraries(tflite_kat PRIVATE tensorflow-lite)
  set_target_properties(tflite_kat PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${MAYHEM_OUT})
endif()
