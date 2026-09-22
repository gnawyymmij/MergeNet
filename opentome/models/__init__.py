"""Paper-facing MergeNet registrations included in the ImageNet delivery.

The matched DeiT baseline is provided by the pinned upstream timm package.
"""

from opentome.models.mergenet import CLSHybridToMeModel, mergenet_small_cls

__all__ = ["CLSHybridToMeModel", "mergenet_small_cls"]
