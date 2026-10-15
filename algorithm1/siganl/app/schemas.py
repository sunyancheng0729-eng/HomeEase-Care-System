# -*- coding: utf-8 -*-
"""请求 / 响应数据模型。"""
from typing import Optional
from pydantic import BaseModel, Field


class SimulateRequest(BaseModel):
    """模拟数据输入（整段离线检测）请求。"""
    seq: Optional[str] = Field(None, description="内置 URFD 序列名，如 fall-01")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="报警阈值，缺省用系统默认 0.5")
    save: bool = Field(True, description="是否把检出的跌倒写入数据库")
    include_curve: bool = Field(True, description="是否返回逐窗概率曲线（供前端画检测时间轴）")
    label: Optional[str] = Field(None, description="自定义标签（如上传文件名），仅用于记录备注")


class SamplePush(BaseModel):
    """单个加速度样本（真实传感器接入用）。"""
    t: float
    ax: float
    ay: float
    az: float


class ThresholdUpdate(BaseModel):
    threshold: float = Field(..., ge=0.0, le=1.0)
