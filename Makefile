include plat.mk

CXX ?= g++

SRCS := main.cpp load_files.cpp kernel_naive.cpp kernel_pq.cpp timer.cpp kernel_utils.cpp
OBJS := $(SRCS:%.cpp=objs/%.o)
DEPS := $(SRCS:%.cpp=deps/%.d)

CXXFLAGS := -O3 -std=c++20 -g -march=znver2
CPPFLAGS :=
LDFLAGS := -g

ifeq ($(PLATFORM),win32)
	EXESUFF := .exe
	EXTRA_OBJS := plat_win32/manifest.o
	CPPFLAGS += -I"$(HOME)/Downloads/Tools/blis-5.0/include/zen2"
	LDFLAGS += -L"$(HOME)/Downloads/Tools/blis-5.0/lib/zen2"
	LIBS := -lblis
else
	EXESUFF :=
	LIBS := -lblas
endif

test$(EXESUFF): objs/main.o objs/load_files.o objs/kernel_naive.o objs/kernel_pq.o objs/timer.o objs/kernel_utils.o $(EXTRA_OBJS)
	$(CXX) -o $@ $(LDFLAGS) $^ $(LIBS)

objs/%.o: %.cpp
	$(CXX) -c -o $@ $(CXXFLAGS) $(CPPFLAGS) $< -MMD -MP -MF deps/$*.d

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
	rmdir detect

clean:
	$(RM) $(OBJS) $(DEPS) $(EXTRA_OBJS) plat.mk

.PHONY: clean
