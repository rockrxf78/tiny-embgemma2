/*
 * SPDX-FileCopyrightText: Copyright 2021, 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
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
 */
// Auto-generated file
// ** DO NOT EDIT **

#ifndef TIMING_ADAPTER_SETTINGS_H
#define TIMING_ADAPTER_SETTINGS_H

#define TA_SRAM_BASE   (0x58103000)
#define TA_EXT_BASE    (0x58103200)

/* Timing adapter settings for SRAM */
#if defined(TA_SRAM_BASE)

#define SRAM_MAXR           (8)
#define SRAM_MAXW           (8)
#define SRAM_MAXRW          (0)
#define SRAM_RLATENCY       (32)
#define SRAM_WLATENCY       (32)
#define SRAM_PULSE_ON       (3999)
#define SRAM_PULSE_OFF      (1)
#define SRAM_BWCAP          (4000)
#define SRAM_PERFCTRL       (0)
#define SRAM_PERFCNT        (0)
#define SRAM_MODE           (1)
#define SRAM_HISTBIN        (0)
#define SRAM_HISTCNT        (0)

#endif /* defined(TA_SRAM_BASE) */

/* Timing adapter settings for EXT */
#if defined(TA_EXT_BASE)

#define EXT_MAXR           (2)
#define EXT_MAXW           (0)
#define EXT_MAXRW          (0)
#define EXT_RLATENCY       (64)
#define EXT_WLATENCY       (0)
#define EXT_PULSE_ON       (320)
#define EXT_PULSE_OFF      (80)
#define EXT_BWCAP          (50)
#define EXT_PERFCTRL       (0)
#define EXT_PERFCNT        (0)
#define EXT_MODE           (1)
#define EXT_HISTBIN        (0)
#define EXT_HISTCNT        (0)

#endif /* defined(TA_EXT_BASE) */

#endif /* TIMING_ADAPTER_SETTINGS_H */
