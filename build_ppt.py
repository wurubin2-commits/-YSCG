"""生成 YSCG Harness 方案汇报 PPT（16 页，含 3 张图形化流程图）。"""
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn

# ---------- 主题 ----------
NAVY = RGBColor(0x0F, 0x2A, 0x44)
TEAL = RGBColor(0x0E, 0x7C, 0x86)
ORANGE = RGBColor(0xE8, 0x73, 0x3A)
RED = RGBColor(0xC0, 0x39, 0x2B)
GRAY = RGBColor(0x5A, 0x6B, 0x7B)
LIGHT = RGBColor(0xF2, 0xF5, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Microsoft YaHei"

W, H = 13.333, 7.5


def _font(run, size, bold=False, color=NAVY, italic=False):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = FONT
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", FONT)


def _bg(slide, color):
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(W), Inches(H))
    rect.fill.solid(); rect.fill.fore_color.rgb = color
    rect.line.fill.background(); rect.shadow.inherit = False
    sp = rect._element
    sp.getparent().remove(sp)
    slide.shapes._spTree.insert(2, sp)


def _title(slide, text, sub=None):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(0.55), Inches(0.09), Inches(0.62))
    bar.fill.solid(); bar.fill.fore_color.rgb = ORANGE; bar.line.fill.background()
    tb = slide.shapes.add_textbox(Inches(0.8), Inches(0.42), Inches(11.8), Inches(0.8))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = text
    _font(r, 25, True, NAVY)
    if sub:
        tb2 = slide.shapes.add_textbox(Inches(0.8), Inches(1.0), Inches(11.8), Inches(0.4))
        p2 = tb2.text_frame.paragraphs[0]
        r2 = p2.add_run(); r2.text = sub
        _font(r2, 12, False, GRAY)


def _footer(slide, idx):
    tb = slide.shapes.add_textbox(Inches(0.6), Inches(7.05), Inches(9), Inches(0.3))
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = "YSCG · 工业模型自进化系统 ｜「智训引擎」赛题"
    _font(r, 9, False, GRAY)
    tb2 = slide.shapes.add_textbox(Inches(12.2), Inches(7.05), Inches(0.8), Inches(0.3))
    p = tb2.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    r2 = p.add_run(); r2.text = str(idx)
    _font(r2, 10, False, GRAY)


def _bullets(slide, items, top=1.55, left=0.85, width=11.7, height=5.0, size=16, gap=10):
    tf = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height)).text_frame
    tf.word_wrap = True
    first = True
    for item in items:
        if isinstance(item, tuple):
            text, level = item
        else:
            text, level = item, 0
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(gap); p.line_spacing = 1.18
        if "｜" in text:
            lead, rest = text.split("｜", 1)
            r1 = p.add_run(); r1.text = ("• " if level == 0 else "   – ") + lead + "｜"
            _font(r1, size - (2 if level else 0), True, TEAL if level == 0 else GRAY)
            r2 = p.add_run(); r2.text = rest
            _font(r2, size - (2 if level else 0), False, NAVY)
        else:
            r = p.add_run(); r.text = ("• " if level == 0 else "   – ") + text
            _font(r, size - (2 if level else 0), False, NAVY if level == 0 else GRAY)


def _table(slide, data, left=0.85, top=1.75, width=11.7, height=4.6, fs=(13, 12)):
    rows, cols = len(data), len(data[0])
    tbl = slide.shapes.add_table(rows, cols, Inches(left), Inches(top), Inches(width), Inches(height)).table
    for i, row in enumerate(data):
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = ""
            cell.margin_left = Inches(0.08); cell.margin_right = Inches(0.08)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            r = p.add_run(); r.text = str(val)
            _font(r, fs[0] if i == 0 else fs[1], i == 0, WHITE if i == 0 else NAVY)
            cell.fill.solid()
            cell.fill.fore_color.rgb = TEAL if i == 0 else (LIGHT if i % 2 == 0 else WHITE)


def _card(slide, left, top, w, h, title, desc):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(w), Inches(h))
    box.adjustments[0] = 0.08
    box.fill.solid(); box.fill.fore_color.rgb = LIGHT
    box.line.color.rgb = TEAL; box.line.width = Pt(1.0)
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = Inches(0.15); tf.margin_right = Inches(0.15); tf.margin_top = Inches(0.1)
    p = tf.paragraphs[0]
    r = p.add_run(); r.text = title
    _font(r, 15, True, TEAL)
    p2 = tf.add_paragraph()
    r2 = p2.add_run(); r2.text = desc
    _font(r2, 11.5, False, NAVY)


# ---------- 流程图元件 ----------
def _shape(slide, kind, l, t, w, h, fill, text=None, color=WHITE, size=12, bold=True, line=None):
    shp = slide.shapes.add_shape(kind, Inches(l), Inches(t), Inches(w), Inches(h))
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line; shp.line.width = Pt(1.0)
    shp.shadow.inherit = False
    if text:
        tf = shp.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.04); tf.margin_right = Inches(0.04)
        tf.margin_top = Inches(0.02); tf.margin_bottom = Inches(0.02)
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        lines = text.split("\n")
        for i, ln in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = PP_ALIGN.CENTER
            r = p.add_run(); r.text = ln
            _font(r, size, bold, color)
    return shp


def _arrow_down(slide, cx, y, h=0.15, color=ORANGE, w=0.16):
    _shape(slide, MSO_SHAPE.DOWN_ARROW, cx - w / 2, y, w, h, color)


def _arrow_right(slide, x, cy, w=0.22, color=ORANGE, h=0.16):
    _shape(slide, MSO_SHAPE.RIGHT_ARROW, x, cy - h / 2, w, h, color)


def _text(slide, l, t, w, h, text, size=11, color=GRAY, bold=False, align=PP_ALIGN.LEFT):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = ln
        _font(r, size, bold, color)
    return tb


def main():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    blank = prs.slide_layouts[6]
    idx = 0

    def nxt():
        nonlocal idx
        idx += 1
        return prs.slides.add_slide(blank)

    # ---------- 1 封面 ----------
    s = nxt(); _bg(s, NAVY)
    accent = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(2.35), Inches(0.12), Inches(1.15))
    accent.fill.solid(); accent.fill.fore_color.rgb = ORANGE; accent.line.fill.background()
    _text(s, 1.2, 2.1, 11, 1.0, "YSCG 工业模型自进化系统", 40, WHITE, True)
    _text(s, 1.2, 3.2, 11, 0.6, "「智训引擎」赛题 Harness 方案汇报", 20, RGBColor(0xC9, 0xD6, 0xE2))
    _text(s, 1.2, 5.6, 11, 0.5, "漂移监测 → 诊断归因 → 实验优化 → 评估上线 / 回滚   ｜   受控 RSI",
          13, RGBColor(0x9F, 0xB3, 0xC6))
    _text(s, 1.2, 6.3, 11, 0.4, "2026.11", 12, RGBColor(0x9F, 0xB3, 0xC6))

    # ---------- 2 目录 ----------
    s = nxt(); _title(s, "目录")
    _bullets(s, [
        "01  项目背景：赛题与痛点", "02  赛题规则要点（Harness SDK 协议）",
        "03  三个工业任务与官方基线", "04  总体方案：两层自进化（受控 RSI）",
        "05  闭环主流程（图）", "06  一轮训练 9 步（图）",
        "07  准入决策 5 道门（图）", "08  稳定性规则与回滚护栏",
        "09  创新点", "10  实验结果①：固定点问题",
        "11  实验结果②：A+ 与漂移修复", "12  当前进度与差距",
        "13  路线图与里程碑",
    ], top=1.55, size=14.5, gap=6)
    _footer(s, idx)

    # ---------- 3 背景 ----------
    s = nxt(); _title(s, "项目背景：赛题与痛点")
    _bullets(s, [
        "赛题｜「智训引擎」工业制造场景下的模型训练与超参数优化",
        "痛点｜工业模型会因传感器老化、设备磨损、工况变化而漂移，性能持续下降",
        "现状｜传统依赖工程师反复处理新数据、新标签，缺乏长期自主迭代能力",
        "目标｜交付能自己干活的 Harness，无人干预完成五步闭环：监测→诊断→实验→评估→上线/回滚",
        "硬门槛｜SwanLab 公开记录、≥3 次迭代、超过官方静态基线、镜像 ≤2GB 且离线/无 GPU",
    ], top=1.6, size=16, gap=12)
    _footer(s, idx)

    # ---------- 4 规则要点 ----------
    s = nxt(); _title(s, "赛题规则要点（Harness SDK 协议）")
    _table(s, [
        ["要求", "内容"],
        ["固定入口", "describe / train / predict；模块默认 /app/contest_submission.py"],
        ["目录权限", "/app、/data、/previous、/state 只读；/workspace 可写；/model 仅预测时挂载"],
        ["每轮强制", "至少 1 次真实模型调用；finish 必须带非空 model_call_ids"],
        ["候选限制", "≤64 个普通文件 / 合计 64MiB；hypothesis/change_summary ≤4000B"],
        ["预测协议", "一行进一行出、每行 flush；分类 label 唯一/最高分唯一；回归非负"],
        ["安全红线", "/app 只读、/previous 不改字节；密钥只填平台、不入镜像"],
    ], top=1.55, height=4.8, fs=(13, 12))
    _footer(s, idx)

    # ---------- 5 三任务与基线 ----------
    s = nxt(); _title(s, "三个工业任务与官方基线", "CPU NumPy MLP，参数量 < 100k；下列为 public validation 实测")
    _table(s, [
        ["数据集", "任务", "主指标", "官方基线"],
        ["A · 气体传感器", "6 类分类", "macro-F1 ↑", "0.7140（acc 0.8786）"],
        ["B · 轴承退化", "RUL 回归（分钟）", "RMSE ↓", "681.16（MAE 563.11）"],
        ["C · CMAPSS", "RUL 回归（循环）", "RMSE ↓", "52.87（MAE 38.90）"],
    ], top=1.7, height=2.5, fs=(14, 13))
    _bullets(s, [
        "三任务覆盖｜分类漂移 + 回归退化，对应赛题“三个正交任务兼顾泛化”",
        "基线用途｜作为准入门对照、回滚兜底与增益基准",
    ], top=4.6, size=15, gap=10)
    _footer(s, idx)

    # ---------- 6 两层自进化 ----------
    s = nxt(); _title(s, "总体方案：两层自进化（受控 RSI）")
    _bullets(s, [
        "L0 确定性主干｜规则状态机，保证无人干预、离线可跑、可复现",
        "L1 模型层自迭代｜模型池 + 受限预算 HPO；漂移触发增量更新 / 重训",
        "L2 策略层自改进｜阈值、预算、触发策略版本化；回放评估后固化",
        "受控 RSI｜LLM 只做“假设生成 / 策略建议”，动作决策始终由规则引擎兜底",
    ], top=1.6, size=16, gap=13)
    _footer(s, idx)

    # ---------- 7 闭环主流程图 ----------
    s = nxt(); _title(s, "闭环主流程（图）：监测 → 诊断 → 决策 → 实验 → 准入 → 上线/回滚")
    mx, mw = 3.55, 3.4
    cy = mx + mw / 2
    # 节点
    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, mx, 1.28, mw, 0.58, TEAL, "① 漂移监测（PSI + 置信度）", size=12)
    _arrow_down(s, cy, 1.86)
    _shape(s, MSO_SHAPE.DIAMOND, mx + 0.45, 2.01, mw - 0.9, 0.62, ORANGE, "漂移且受损？", size=10.5, color=WHITE)
    _arrow_down(s, cy, 2.63)
    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, mx, 2.78, mw, 0.58, NAVY, "② 诊断归因 → ③ 策略决策", size=11.5)
    _arrow_down(s, cy, 3.36)
    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, mx, 3.51, mw, 0.58, TEAL, "④ 实验优化（增量 / 重训 / HPO）", size=11.5)
    _arrow_down(s, cy, 4.09)
    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, mx, 4.24, mw, 0.58, NAVY, "⑤ 准入评估（增益→锚点→记忆）", size=11.5)
    _arrow_down(s, cy, 4.82)
    _shape(s, MSO_SHAPE.DIAMOND, mx + 0.35, 4.97, mw - 0.7, 0.66, ORANGE, "全部门通过且可回滚？", size=9.5, color=WHITE)
    _arrow_down(s, cy, 5.63)
    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, mx, 5.78, mw, 0.58, RGBColor(0x1F, 0x8A, 0x5A), "⑥ 上线（champion 替换）", size=12)
    # 侧分支
    _arrow_right(s, 6.95, 2.32, 0.85)
    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 7.8, 2.0, 4.4, 0.66, LIGHT,
           "维持（NOOP）\n漂移但模型健康 → 省算力", size=11, color=NAVY, line=GRAY)
    _arrow_right(s, 6.95, 5.3, 0.85)
    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 7.8, 4.97, 4.4, 0.66, LIGHT,
           "回滚 / 归档\n记录失败原因码，champion 不变", size=11, color=NAVY, line=GRAY)
    # 回流说明
    _text(s, 0.7, 6.55, 12, 0.4, "↺ 上线 / 回滚后回到 ① 漂移监测，形成无人干预闭环", 13, TEAL, True)
    _footer(s, idx)

    # ---------- 8 一轮训练流程图 ----------
    s = nxt(); _title(s, "一轮训练（图）：6 个阶段，9 个动作")
    phases = [
        ("① 输入", "读 /data 训练/验证 JSONL\n读父模型 rnd.parent\n读 /state 跨轮状态"),
        ("② 假设", "诊断本轮漂移\n（输入 PSI + 置信度）\nLLM 1 次真实调用"),
        ("③ 训练", "从父模型增量续训\n保留 Adam / 标准化\n（HPO 兜底）"),
        ("④ 自检", "锚点检查\n记忆检查\n验证器自检"),
        ("⑤ 提交", "submit_candidate\n候选 ≤64 文件/64MiB\n平台测量返回"),
        ("⑥ 收尾", "决策：新/父/回滚\nsave_state\nfinish(model_call_ids)"),
    ]
    bx, bw, gap = 0.7, 1.75, 0.28
    for i, (h, desc) in enumerate(phases):
        x = bx + i * (bw + gap)
        _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.65, bw, 0.85,
               TEAL if i % 2 == 0 else NAVY, h, size=13)
        if i < len(phases) - 1:
            _arrow_right(s, x + bw + 0.03, 2.07, 0.22)
        _text(s, x, 2.65, bw, 1.6, desc, 9.5, GRAY, False, PP_ALIGN.CENTER)
    # 底部硬约束
    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.7, 4.55, 11.93, 0.85, LIGHT,
           "每轮硬约束：≥1 次真实模型调用 ｜ 候选 ≤64 文件 / 64MiB ｜ finish 带非空 model_call_ids",
           size=12.5, color=NAVY, line=TEAL)
    _text(s, 0.7, 5.65, 11.93, 0.9,
          "决策点：平台返回测量 + rnd.history() 决定 选新候选 / 保留父模型 / 回滚更早产物",
          12, GRAY)
    _footer(s, idx)

    # ---------- 9 准入决策 5 道门 ----------
    s = nxt(); _title(s, "准入决策（图）：5 道门，任一失败即回滚/归档")
    gates = ["候选", "G1\n增益", "G2\n锚点", "G3\n记忆", "G4\n验证器", "G5\n回滚", "上线"]
    gx, gw, ggap = 0.6, 1.6, 0.19
    centers = []
    for i, g in enumerate(gates):
        x = gx + i * (gw + ggap)
        fill = RGBColor(0x1F, 0x8A, 0x5A) if i == len(gates) - 1 else (GRAY if i == 0 else (TEAL if i % 2 else NAVY))
        _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, 1.8, gw, 0.95, fill, g, size=11.5)
        centers.append(x + gw / 2)
        if i < len(gates) - 1:
            _arrow_right(s, x + gw + 0.02, 2.27, 0.15)
    # 失败箭头
    for c in centers[1:-1]:
        _arrow_down(s, c, 2.75, 0.35, RED)
    _shape(s, MSO_SHAPE.ROUNDED_RECTANGLE, 0.6, 3.15, 12.34, 0.7, RGBColor(0xFB, 0xE9, 0xE7),
           "任一失败 → 归档为 failed challenger，champion 不变，记录失败原因码（gate 编码）",
           size=12.5, color=RED, line=RED)
    # 门说明
    _bullets(s, [
        "G1 增益｜挑战者窗口准确率 ≥ 冠军 + min_gain",
        "G2 锚点｜每维 ≥ 历史 floor − ε（防能力退化，floor 只升不降）",
        "G3 记忆｜分层回归样本上回归计数 ≤ 容差",
        "G4 验证器｜好/坏候选排序正确；失效则拒绝本轮所有候选",
        "G5 回滚｜旧版本完整保留 + 重载预测一致（回滚预演）",
    ], top=4.05, size=13, gap=6)
    _footer(s, idx)

    # ---------- 10 稳定性规则 ----------
    s = nxt(); _title(s, "核心机制：稳定性规则（锚点 + 记忆）")
    _bullets(s, [
        "锚点检查｜冻结切片分数有单调不减的 floor（棘轮），候选不得低于历史最优",
        "记忆检查｜容量受限的保护性记录集，只存“已验证正确”的样本",
        "分层回归样本｜覆盖 场景 × 难度 × 指标面，回归计数 ≤ 容差才放行",
        "设计动机｜防“新分布涨分、旧能力被悄悄丢掉”的假进步",
    ], top=1.6, size=16, gap=13)
    _footer(s, idx)

    # ---------- 11 验证器 + 回滚 ----------
    s = nxt(); _title(s, "核心机制：验证器自检 + 回滚护栏")
    _bullets(s, [
        "验证器自检｜用已知好/坏候选检验排序；排序错误 → 拒绝本轮所有候选",
        "回滚护栏｜旧版本完整保留 + 重载预测一致（回滚预演）",
        "上线条件｜只有“可安全回滚”的候选才允许进入无人工干预上线",
        "失败处理｜任一门失败 → 归档并记录失败原因码，champion 不变",
    ], top=1.6, size=16, gap=13)
    _footer(s, idx)

    # ---------- 12 创新点 ----------
    s = nxt(); _title(s, "创新点")
    cards = [
        ("受控 RSI", "规则兜底 + LLM 参谋，\nLLM 永不直接改权重"),
        ("锚点棘轮", "能力下限单调不减，\n防退化有数学保证"),
        ("分层记忆回归", "多维覆盖，\n防历史能力遗忘"),
        ("验证器自检", "验证器失效即熔断，\n拒绝本轮所有候选"),
        ("回滚预演", "上线前先验证退路，\n回滚不是免费的"),
        ("成本受控", "默认不更新 + 预算硬顶，\n增益/成本导向"),
    ]
    cw, ch = 3.55, 1.75
    for i, (t, d) in enumerate(cards):
        col, row = i % 3, i // 3
        _card(s, 0.85 + col * (cw + 0.35), 1.7 + row * (ch + 0.45), cw, ch, t, d)
    _footer(s, idx)

    # ---------- 13 实验结果①：固定点 ----------
    s = nxt(); _title(s, "实验结果①：固定点问题（原始 10 轮对照组）",
                     "同一 champion + 同一数据/种子/epoch → 第 3 轮后候选完全相同")
    _table(s, [
        ["数据集", "起始", "10 轮最佳", "接受轮次", "现象"],
        ["A（分类）", "0.7140", "0.7264", "4/10", "第 5 轮起候选重复，准入全部拒绝"],
        ["B（回归）", "681.16", "647.24", "2/10", "第 3 轮起候选完全相同"],
        ["C（回归）", "52.87", "51.93", "2/10", "第 3 轮起候选完全相同"],
    ], top=1.6, height=2.6, fs=(12, 11.5))
    _bullets(s, [
        "准入正确｜重复/退化候选被拒绝，champion 不变（回滚保护生效）",
        "搜索停滞｜后续轮次不再产生新信息 → “多轮 ≠ 进化”",
        "结论｜必须注入有效多样性，并闭环“拒绝反馈”",
    ], top=4.65, size=14.5, gap=9)
    _footer(s, idx)

    # ---------- 14 实验结果②：A+ 与漂移修复 ----------
    s = nxt(); _title(s, "实验结果②：A+ 多样性 + 漂移修复（实验组）",
                     "候选唯一率 100% · 连续重复 0 · best 不降 · 回滚保护仍生效")
    _table(s, [
        ["A+ 多样性", "起始", "10 轮最佳", "接受", "唯一率", "连续重复"],
        ["A", "0.7264", "0.7264", "0/10", "100%", "0"],
        ["B", "647.24", "647.24", "0/10", "100%", "0"],
        ["C", "51.93", "51.66", "3/10", "100%", "0"],
    ], top=1.45, height=1.9, fs=(11.5, 11))
    _table(s, [
        ["B 漂移", "起始", "PSI 触发", "接受", "best", "改善"],
        ["A", "0.7264", "9/10", "1", "0.7269", "+0.0005"],
        ["B", "647.24", "9/10", "1", "627.95", "−19.29"],
        ["C", "51.93", "9/10", "3", "51.68", "−0.25"],
    ], top=3.55, height=1.9, fs=(11.5, 11))
    _bullets(s, [
        "固定点消除｜配置增量（lr / 类权重 / 重采样 / 正则 / recency / 集成）→ 候选 100% 唯一",
        "同分布饱和｜A/B 在 A+ 中 0 接受 → 需要新的数据信号",
        "漂移触发有效｜B −19.29、C −0.25；变差重训被回滚保留 champion",
    ], top=5.75, size=12.5, gap=5)
    _footer(s, idx)

    # ---------- 14 进度与差距 ----------
    s = nxt(); _title(s, "当前进度与差距")
    _table(s, [
        ["模块", "状态"],
        ["算法内核（监测 / 决策 / 准入 / 回滚）", "✅ 完成，真实数据跑通"],
        ["协议骨架 contest_submission", "✅ 骨架 + 本地演练"],
        ["三基线接入 + SwanLab 记录", "✅ 完成"],
        ["官方 SDK / harness.py", "❌ 未接入"],
        ["Docker 离线镜像（≤2GB）", "❌ 未制作"],
        ["真实 LLM 调用（千问 4B/27B）", "❌ 未接入"],
    ], top=1.55, height=4.3, fs=(13, 12))
    _footer(s, idx)

    # ---------- 15 路线图 ----------
    s = nxt(); _title(s, "路线图与里程碑")
    _bullets(s, [
        "P0 硬门槛｜接 SDK 与 /app/harness.py → Docker 镜像 ≤2GB → 三任务跑通 → 超过静态基线",
        "P1 分数提升｜千问 4B 顾问、变点检测、策略层回放自改进、对照试验",
        "P2 材料｜实验汇总表、消融实验、成本收益分析、答辩 PPT",
        "时间线｜11/13 校赛推荐截止 → 11/15–12/5 企业赛 → 12/26 总决赛",
    ], top=1.6, size=16, gap=12)
    _footer(s, idx)

    # ---------- 16 结束 ----------
    s = nxt(); _bg(s, NAVY)
    _text(s, 0.9, 2.6, 11.5, 1.0, "谢谢！", 44, WHITE, True)
    _text(s, 0.9, 3.85, 11.5, 0.6, "YSCG · 工业模型自进化系统   ｜   可无人干预、可回滚、可留痕",
          16, RGBColor(0xC9, 0xD6, 0xE2))

    out = "YSCG_Harness_方案汇报.pptx"
    prs.save(out)
    print("saved:", out, "| slides:", len(prs.slides))


if __name__ == "__main__":
    main()
