include plat.mk

CXX ?= g++

SRCS := main.cpp load_files.cpp kernel_naive.cpp kernel_pq.cpp kernel_hnsw.cpp kernel_vamana.cpp timer.cpp kernel_utils.cpp simd_utils.cpp run_filtered.cpp
OBJS := $(SRCS:%.cpp=objs/%.o) $(SRCS:%.cpp=objs_debug/%.o)
DEPS := $(SRCS:%.cpp=deps/%.d) $(SRCS:%.cpp=deps_debug/%.d)

CXXFLAGS := -std=c++20 -Wall -Wextra -fopenmp
CXXFLAGS_RELEASE := -O3 -g -march=native
CPPFLAGS_RELEASE := -DNDEBUG
CXXFLAGS_DEBUG := -g -march=native
CPPFLAGS_DEBUG :=
LDFLAGS := -g -fopenmp

ifeq ($(PLATFORM),win32)
	EXESUFF := .exe
	EXTRA_OBJS := plat_win32/manifest.o
	CPPFLAGS += -isystem "$(HOME)/Downloads/Tools/blis-5.0/include/zen2"
	LDFLAGS += -L"$(HOME)/Downloads/Tools/blis-5.0/lib/zen2"
	CPPFLAGS +=
	LIBS := -lblis-mt
else ifeq ($(BLAS),mkl)
	EXESUFF :=
	CPPFLAGS +=
	LIBS := -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
else
	EXESUFF :=
	CPPFLAGS +=
	LIBS := -lopenblas
endif

EXE_NAMES := test run_filtered
EXES := $(foreach exe,$(EXE_NAMES),$(exe)$(EXESUFF))
EXES_DEBUG := $(foreach exe,$(EXE_NAMES),$(exe)_debug$(EXESUFF))

test_OBJS := main.o load_files.o kernel_naive.o kernel_pq.o kernel_hnsw.o kernel_vamana.o timer.o kernel_utils.o simd_utils.o
run_filtered_OBJS := run_filtered.o load_files.o timer.o kernel_utils.o simd_utils.o

all: $(EXES)

all_debug: $(EXES_DEBUG)

test$(EXESUFF): $(foreach file,$(test_OBJS),objs/$(file)) $(EXTRA_OBJS)
	$(CXX) -o $@ $(LDFLAGS) $^ $(LIBS)

test_debug$(EXESUFF): $(foreach file,$(test_OBJS),objs_debug/$(file)) $(EXTRA_OBJS)
	$(CXX) -o $@ $(LDFLAGS) $^ $(LIBS)

run_filtered$(EXESUFF): $(foreach file,$(run_filtered_OBJS),objs/$(file)) $(EXTRA_OBJS)
	$(CXX) -o $@ $(LDFLAGS) $^ $(LIBS)

run_filtered_debug$(EXESUFF): $(foreach file,$(run_filtered_OBJS),objs_debug/$(file)) $(EXTRA_OBJS)
	$(CXX) -o $@ $(LDFLAGS) $^ $(LIBS)

objs/%.o: %.cpp
	$(CXX) -c -o objs/$*.o $(CXXFLAGS_RELEASE) $(CXXFLAGS) $(CPPFLAGS_RELEASE) $(CPPFLAGS) $< -MMD -MP -MF deps/$*.d

objs_debug/%.o: %.cpp
	$(CXX) -c -o objs_debug/$*.o $(CXXFLAGS_DEBUG) $(CXXFLAGS) $(CPPFLAGS_DEBUG) $(CPPFLAGS) $< -MMD -MP -MF deps_debug/$*.d

-include $(DEPS)

plat_win32/manifest.o: plat_win32/manifest.rc plat_win32/manifest.xml
	windres $< -o $@

plat.mk:
	mkdir -p detect
	echo "#if _WIN32" >detect/test_win32.cpp
	echo '# error "we are on windows"' >>detect/test_win32.cpp
	echo "#endif" >>detect/test_win32.cpp
	echo "int main() {}" >>detect/test_win32.cpp
	$(CXX) -o /dev/null detect/test_win32.cpp 2>/dev/null 1>&2 && echo 'PLATFORM := linux' >plat.mk || echo 'PLATFORM := win32' >plat.mk
	rm detect/test_win32.cpp
	echo "#include <mkl.h>" >detect/test_mkl.cpp
	echo "int main() {}" >>detect/test_mkl.cpp
	-$(CXX) -o /dev/null detect/test_mkl.cpp -lmkl_core 2>/dev/null 1>&2 && echo 'BLAS := mkl' >>plat.mk
	rm detect/test_mkl.cpp
	rmdir detect

clean:
	$(RM) $(OBJS) $(DEPS) $(EXTRA_OBJS) plat.mk $(EXES) $(EXES_DEBUG)

.PHONY: clean
.SUFFIXES:
