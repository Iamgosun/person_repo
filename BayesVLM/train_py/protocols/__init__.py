from __future__ import annotations

from train_py.protocols.base_protocol import BaseProtocol
from train_py.protocols.id_protocol import IDProtocol
from train_py.protocols.base2new_protocol import Base2NewProtocol
from train_py.protocols.xd_protocol import XDProtocol
from train_py.protocols.dg_protocol import DGProtocol


def build_protocol(protocol_name: str) -> BaseProtocol:
    key = str(protocol_name).strip().lower()
    if key == "id":
        return IDProtocol()
    if key == "base2new":
        return Base2NewProtocol()
    if key == "xd":
        return XDProtocol()
    if key == "dg":
        return DGProtocol()
    raise ValueError(f"unknown protocol: {protocol_name}")
