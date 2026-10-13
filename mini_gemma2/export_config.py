"""把最初的 Python 定义（intents.py 的意图/模板/槽位 + 口头前后缀 + ASR 噪声默认参数）导出成默认 YAML 配置。

只在需要“从 Python 定义重新生成默认配置”时使用；日常改数据请直接编辑 configs/*.yaml。
用法: python mini_gemma2/export_config.py [--out configs/smarthome_zh.yaml]
"""
import argparse
import dataclasses
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_gemma2 import intents as I
from mini_gemma2.noise import AsrNoiseConfig

# 口头前后缀（最初在 build_dataset.py 里硬编码）
PREFIX = ["帮我", "请", "麻烦", "你帮我", "给我", "小智小智", "小智", "嗨小智", "那个", "快点", "赶紧", "现在", "马上",
          "哎", "诶", "嗯", "好的", "劳驾", "你好", "喂", "对了", "呃", "小家小家", "嘿"]
PREFIX_TEST_EXTRA = ["小智你好", "天啊", "麻烦你", "唉", "喂喂"]
SUFFIX = ["吧", "啊", "呀", "呢", "谢谢", "好吗", "行吗", "可以吗", "快点", "马上", "现在", "立刻", "一下", "哦", "啦", "呗"]
SUFFIX_TEST_EXTRA = ["谢啦", "可以不", "好不好", "行不行", "拜托了"]
NO_PREFIX_START = ["我", "帮", "麻", "给", "请", "让", "来", "你", "能"]
NO_SUFFIX_END = ["吗", "吧", "呢", "啊", "呀", "啦", "哦", "呗"]

HEADER = """# 智能家居意图识别数据生成配置（中文）。
# 这是数据生成的唯一事实来源：改意图措辞/加模板/改 ASR 错字规则都改这里，不用动 Python。
# 生成:      python mini_gemma2/build_dataset.py --config configs/smarthome_zh.yaml --out data
# 校验:      python mini_gemma2/check_dataset.py --config configs/smarthome_zh.yaml --lock data/intents.json
#
# 规则速记
#  * 模板占位符: {R}房间 {D}设备别名 {N}空调温度 {W}水温 {P}百分比，以及 slots 里的任意大写键（如 {COLOR}）
#  * 每个意图的 train 列表里，**最后 n_dev_templates 条会被留出做 dev**（训练不会见到），所以新增训练模板请插在它们之前
#  * test 模板必须和 train 的措辞不同（评测测的是“没见过的说法”）；test 只用于最终报告
#  * 意图的顺序和 id 决定下游 C 库里 sh_intent_id[] 的下标；要保持兼容就不要增删/重排（check_dataset.py --lock 会检查）
"""


def section(comment, data):
    txt = yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=None, width=140)
    return f"\n# ---------------------------------------------------------------- {comment}\n{txt}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "smarthome_zh.yaml"))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    devices = {}
    for key, (name, alias, alias_test, on_x, on_xt, off_x, off_xt) in I.POWER_DEVICES.items():
        d = {"name": name, "alias": alias}
        for k, v in (("alias_test_extra", alias_test), ("on_extra_train", on_x), ("on_extra_test", on_xt),
                     ("off_extra_train", off_x), ("off_extra_test", off_xt)):
            if v:
                d[k] = v
        devices[key] = d
    n_power = 2 * len(I.POWER_DEVICES)
    explicit = [{"id": it.id, "group": it.group, "std": it.std, "train": it.train, "test": it.test}
                for it in I.build_intents()[n_power:]]

    asr = dataclasses.asdict(AsrNoiseConfig())
    asr = {k: (list(v) if isinstance(v, tuple) else v) for k, v in asr.items()}

    out = HEADER
    out += section("槽位候选：模板里的 {COLOR} {CH} {SONG} {T} {NAME} {PLACE}（渲染按此顺序替换；可新增任意大写键）", {
        "slots": {"COLOR": I.COLORS, "CH": I.CHANNELS, "SONG": I.SONGS, "T": I.DURATIONS, "NAME": I.NAMES, "PLACE": I.PLACES}})
    out += section("房间：{R}。test 可以比 train 多出没见过的房间", {"rooms": {"train": I.ROOMS_TRAIN, "test": I.ROOMS_TEST}})
    out += section("数值槽位：{N} 空调温度、{W} 水温（闭区间）、{P} 百分比候选；每个数值随机用阿拉伯数字或中文数字写法",
                   {"numbers": {"N": [16, 30], "W": [35, 60], "percent": list(range(10, 101, 10))}})
    out += section("口头前后缀与标点。test 额外使用 *_test_extra（训练没见过）；以 no_prefix_start 中的字开头的句子不再加前缀，以 no_suffix_end 结尾的不再加后缀",
                   {"wrappers": {"prefix": PREFIX, "prefix_test_extra": PREFIX_TEST_EXTRA, "suffix": SUFFIX,
                                 "suffix_test_extra": SUFFIX_TEST_EXTRA, "no_prefix_start": NO_PREFIX_START,
                                 "no_suffix_end": NO_SUFFIX_END, "question_endings": ["吗", "呢", "没", "少", "几", "么", "样"],
                                 "punct_declarative": ["。", "！"], "p_prefix": 0.45, "p_suffix": 0.35, "p_punct": 0.12}})
    out += section("“开/关设备”这一族（每个设备生成 <key>.on / <key>.off 两个意图）：通用模板 + 每个设备的别名与额外模板。\n"
                   "# 标准指令 = on_std/off_std 里的 {name} 替换成设备 name",
                   {"power": {"on_std": "打开{name}", "off_std": "关闭{name}", "on_train": I.ON_TRAIN, "on_test": I.ON_TEST,
                              "off_train": I.OFF_TRAIN, "off_test": I.OFF_TEST, "devices": devices}})
    out += section("其余意图。id/group/std/train/test；group 用于分组（挖 hard negative 时参考）。顺序即 intents.json 的顺序",
                   {"intents": explicit})
    out += section("OOD（非家居意图）句子模板：只用于蒸馏与拒识阈值标定，不是意图。train 的最后 n_dev_templates 条留出作 OOD-dev",
                   {"ood": {"n_dev_templates": 8, "n_per_template": {"train": 50, "dev": 40, "test": 12}, "p_typo": 0.4,
                            "train": I.OOD_TRAIN, "test": I.OOD_TEST}})
    out += section("切分与采样", {"split": {"n_dev_templates": 2},
                             "sampling": {"p_room": 0.45, "p_room_de": 0.5, "p_typo": 0.42, "p_std_typo": 0.06},
                             "counts": {"train": 420, "dev": 40, "test": 80}})
    out += section("ASR 错字噪声。custom_confusions 用来放真实 ASR 的错误统计：{正确字: {常被识别成的错字: 权重}}，\n"
                   "# 例如 {开: {凯: 3, 揩: 1}, 灯: {等: 5, 登: 2}}；字命中时以 p_custom 的概率优先用它，否则回落到同音/近音规则",
                   {"asr": asr})
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(out)
    print("wrote", a.out, f"({len(out)} chars)")


if __name__ == "__main__":
    main()
