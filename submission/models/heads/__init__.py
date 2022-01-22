import sys
sys.path.append("models/heads")

from fcn_head import FCNHead
from uper_head import UPerHead
from branch_fpn_head import BranchFPNHead
from unet_decode_head import UnetDecodeHead