/*
 * Copyright © 2026  Vortex GPGPU
 * SPDX-License-Identifier: MIT
 */

#ifndef VP_PUBLIC_H
#define VP_PUBLIC_H

#ifdef __cplusplus
extern "C" {
#endif

struct pipe_screen;
struct sw_winsys;

struct pipe_screen *
vortexpipe_create_screen(struct sw_winsys *winsys);

#ifdef __cplusplus
}
#endif

#endif /* VP_PUBLIC_H */
