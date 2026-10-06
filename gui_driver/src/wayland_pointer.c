#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <wayland-client.h>
#include "wlr-virtual-pointer-client-protocol.h"

static struct zwlr_virtual_pointer_manager_v1 *pointer_mgr = NULL;
static struct wl_seat *seat = NULL;

static void registry_global(void *data, struct wl_registry *registry, uint32_t name, const char *interface, uint32_t version) {
    if (strcmp(interface, zwlr_virtual_pointer_manager_v1_interface.name) == 0) {
        pointer_mgr = wl_registry_bind(registry, name, &zwlr_virtual_pointer_manager_v1_interface, 2);
    } else if (strcmp(interface, "wl_seat") == 0) {
        seat = wl_registry_bind(registry, name, &wl_seat_interface, 1);
    }
}
static void registry_global_remove(void *data, struct wl_registry *registry, uint32_t name) {}
static const struct wl_registry_listener registry_listener = { registry_global, registry_global_remove };

static uint32_t parse_button(const char *btn) {
    if (!btn) return 0x110;
    if (strcmp(btn, "right") == 0) return 0x111;
    if (strcmp(btn, "middle") == 0) return 0x112;
    return 0x110; // left
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <click|click_at|move|down|up|scroll> [args...]\n", argv[0]);
        return 1;
    }

    const char *action = argv[1];

    struct wl_display *display = wl_display_connect(NULL);
    if (!display) {
        fprintf(stderr, "Failed to connect to wayland display\n");
        return 2;
    }

    struct wl_registry *registry = wl_display_get_registry(display);
    wl_registry_add_listener(registry, &registry_listener, NULL);
    wl_display_roundtrip(display);

    if (!pointer_mgr) {
        fprintf(stderr, "zwlr_virtual_pointer_manager_v1 interface not available\n");
        wl_display_disconnect(display);
        return 3;
    }

    struct zwlr_virtual_pointer_v1 *vp = zwlr_virtual_pointer_manager_v1_create_virtual_pointer(pointer_mgr, seat);
    wl_display_roundtrip(display);

    if (strcmp(action, "click") == 0) {
        uint32_t btn = parse_button(argc > 2 ? argv[2] : "left");
        int count = argc > 3 ? atoi(argv[3]) : 1;
        if (count < 1) count = 1;

        for (int i = 0; i < count; i++) {
            zwlr_virtual_pointer_v1_button(vp, 0, btn, 1);
            zwlr_virtual_pointer_v1_frame(vp);
            wl_display_flush(display);
            usleep(30000); // 30ms click duration

            zwlr_virtual_pointer_v1_button(vp, 0, btn, 0);
            zwlr_virtual_pointer_v1_frame(vp);
            wl_display_flush(display);
            if (i + 1 < count) usleep(50000);
        }
    } else if (strcmp(action, "click_at") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s click_at <x> <y> [button] [count] [w] [h]\n", argv[0]);
            return 1;
        }
        uint32_t x = atoi(argv[2]);
        uint32_t y = atoi(argv[3]);
        uint32_t btn = parse_button(argc > 4 ? argv[4] : "left");
        int count = argc > 5 ? atoi(argv[5]) : 1;
        uint32_t w = argc > 6 ? atoi(argv[6]) : 1920;
        uint32_t h = argc > 7 ? atoi(argv[7]) : 1080;

        zwlr_virtual_pointer_v1_motion_absolute(vp, 0, x, y, w, h);
        zwlr_virtual_pointer_v1_frame(vp);
        wl_display_flush(display);
        usleep(20000);

        for (int i = 0; i < count; i++) {
            zwlr_virtual_pointer_v1_button(vp, 0, btn, 1);
            zwlr_virtual_pointer_v1_frame(vp);
            wl_display_flush(display);
            usleep(30000);

            zwlr_virtual_pointer_v1_button(vp, 0, btn, 0);
            zwlr_virtual_pointer_v1_frame(vp);
            wl_display_flush(display);
            if (i + 1 < count) usleep(50000);
        }
    } else if (strcmp(action, "move") == 0) {
        if (argc < 4) {
            fprintf(stderr, "Usage: %s move <x> <y> [w] [h]\n", argv[0]);
            return 1;
        }
        uint32_t x = atoi(argv[2]);
        uint32_t y = atoi(argv[3]);
        uint32_t w = argc > 4 ? atoi(argv[4]) : 1920;
        uint32_t h = argc > 5 ? atoi(argv[5]) : 1080;

        zwlr_virtual_pointer_v1_motion_absolute(vp, 0, x, y, w, h);
        zwlr_virtual_pointer_v1_frame(vp);
        wl_display_flush(display);
    } else if (strcmp(action, "down") == 0) {
        uint32_t btn = parse_button(argc > 2 ? argv[2] : "left");
        zwlr_virtual_pointer_v1_button(vp, 0, btn, 1);
        zwlr_virtual_pointer_v1_frame(vp);
        wl_display_flush(display);
    } else if (strcmp(action, "up") == 0) {
        uint32_t btn = parse_button(argc > 2 ? argv[2] : "left");
        zwlr_virtual_pointer_v1_button(vp, 0, btn, 0);
        zwlr_virtual_pointer_v1_frame(vp);
        wl_display_flush(display);
    } else if (strcmp(action, "scroll") == 0) {
        int delta = argc > 2 ? atoi(argv[2]) : -15; // default scroll down
        // 15 units = standard wheel step; axis 0 is vertical
        wl_fixed_t val = wl_fixed_from_int(delta);
        zwlr_virtual_pointer_v1_axis(vp, 0, 0, val);
        zwlr_virtual_pointer_v1_frame(vp);
        wl_display_flush(display);
    } else {
        fprintf(stderr, "Unknown action: %s\n", action);
        return 1;
    }

    wl_display_roundtrip(display);
    zwlr_virtual_pointer_v1_destroy(vp);
    zwlr_virtual_pointer_manager_v1_destroy(pointer_mgr);
    wl_display_disconnect(display);
    return 0;
}
