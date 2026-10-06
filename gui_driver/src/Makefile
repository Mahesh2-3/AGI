CC ?= gcc
CFLAGS ?= -O2 -Wall
PKG_CONFIG ?= pkg-config

WAYLAND_FLAGS = $(shell $(PKG_CONFIG) --cflags --libs wayland-client)

all: ../bin/wayland_pointer

../bin/wayland_pointer: wayland_pointer.c wlr-virtual-pointer-protocol.c
	mkdir -p ../bin
	$(CC) $(CFLAGS) wayland_pointer.c wlr-virtual-pointer-protocol.c -o ../bin/wayland_pointer $(WAYLAND_FLAGS)

clean:
	rm -f ../bin/wayland_pointer

.PHONY: all clean
