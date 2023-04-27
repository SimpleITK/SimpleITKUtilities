# ========================================================================
#
#  Copyright NumFOCUS
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# ========================================================================

import SimpleITK as sitk
from typing import Sequence, Tuple


def overlay_bounding_boxes(
    image: sitk.Image,
    bounding_boxes: Sequence[Sequence[float]],
    bounding_box_format: str = "MINXY_MAXXY",
    normalized: bool = False,
    colors: Sequence[int] = [],
    half_line_width: int = 0,
) -> Tuple[sitk.Image, bool]:
    """
    Overlay axis aligned bounding boxes on a 2D image. The function supports several ways of specifying a
    bounding box using pixel indexes:

    1. "MINXY_MAXXY" - [min_x, min_y, max_x, max_y]
    2. "MINXY_WH" - [min_x, min_y, width, height]
    3. "CENT_WH" - [center_x, center_y, width, height]
    4. "CENT_HALFWH" - [center_x, center_y, width/2, height/2]

    Bounding boxes are plotted in the order they appear in the iterable/list. To change the overlap between rectangles
    change the order in the list. The last entry in the list will be plotted on top of the previous ones.

    Caveat: When using larger line widths, bounding boxes that are very close to the image border may cause an exception
    and result in partial overlay. A trivial solution is to decrease the value of the half_line_width parameter.

    :param image: Input image, 2D image with scalar or RGB pixels on which we plot the bounding boxes.
    :param bounding_boxes: Bounding boxes to plot. Each bounding box is represented by four numbers.
    :param bounding_box_format: One of ["MINXY_MAXXY", "MINXY_WH", "CENT_WH", "CENT_HALFWH"] specifying the meaning of
    the four entries representing the bounding box.
    :param normalized: Indicate whether the bounding box numbers were normalized to be in [0,1].
    :param colors: Specify the color for each rectangle using RGB values, triplets in [0,255].
    Useful for visually representing different classes (relevant for object detection). Most often a flat
    list, e.g. colors = [255, 0, 0, 0, 255, 0] reresents red, and green.
    :param half_line_width: Plot using thicker lines.
    :return: A tuple where the first entry is a SimpleITK image with rectangles plotted on it and the second entry is a boolean
    which is true if one or more of the rectangles were out of bounds, false otherwise.
    """
    # functions that convert from various bounding box representations to the [min_x, min_y, max_x, max_y] representation.
    convert_to_minxy_maxxy = {
        "MINXY_MAXXY": lambda original: original,
        "MINXY_WH": lambda original: [
            [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            for bbox in original
        ],
        "CENT_WH": lambda original: [
            [
                bbox[0] - bbox[2] / 2.0,
                bbox[1] - bbox[3] / 2.0,
                bbox[0] + bbox[2] / 2.0,
                bbox[1] + bbox[3] / 2.0,
            ]
            for bbox in original
        ],
        "CENT_HALFWH": lambda original: [
            [bbox[0] - bbox[2], bbox[1] - bbox[3], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            for bbox in original
        ],
    }

    # Confirm image is in expected format
    pixel_type = image.GetPixelID()
    num_channels = image.GetNumberOfComponentsPerPixel()
    if pixel_type not in [sitk.sitkUInt8, sitk.sitkVectorUInt8]:
        raise ValueError(
            f"Image channels expected to have type of 8-bit unsigned integer, got ({image.GetPixelIDTypeAsString()})"
        )
    if num_channels not in [1, 3]:
        raise ValueError(
            f"Image expected to have one or three channels, got ({num_channels})"
        )
    if num_channels == 3:
        overlay_image = sitk.Image(image)
    else:
        overlay_image = sitk.Compose([image] * 3)
    if half_line_width < 0:
        raise ValueError(
            f"Half line width parameter expected to be non-negative, got ({half_line_width})"
        )
    # Convert bounding box information into standard format, based on user specification of the original format
    try:
        standard_bounding_boxes = convert_to_minxy_maxxy[bounding_box_format](
            bounding_boxes
        )
        if normalized:
            scale_x, scale_y = image.GetSize()
            standard_bounding_boxes = [
                [
                    bbox[0] * scale_x,
                    bbox[1] * scale_y,
                    bbox[2] * scale_x,
                    bbox[3] * scale_y,
                ]
                for bbox in standard_bounding_boxes
            ]
        # round to integer coordinates
        standard_bounding_boxes = [
            [int(b + 0.5) for b in bbox] for bbox in standard_bounding_boxes
        ]
    except KeyError:
        raise ValueError(
            f"Unknown bounding box format ({bounding_box_format}), valid values are [MINXY_WH, MINXY_MAXXY, CENT_WH, CENT_HALFWH]"
        )
    if not colors:  # use a single color for all bounding boxes
        colors = [[255, 0, 0]] * len(standard_bounding_boxes)
    else:
        colors = [colors[i : i + 3] for i in range(0, len(colors), 3)]
    line_width = 1 + 2 * half_line_width
    out_of_bounds = False
    for bbox, color in zip(standard_bounding_boxes, colors):
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        vert = sitk.Compose(
            [
                sitk.Image([line_width, height + line_width], sitk.sitkUInt8)
                + color[0],
                sitk.Image([line_width, height + line_width], sitk.sitkUInt8)
                + color[1],
                sitk.Image([line_width, height + line_width], sitk.sitkUInt8)
                + color[2],
            ]
        )
        horiz = sitk.Compose(
            [
                sitk.Image([width, line_width], sitk.sitkUInt8) + color[0],
                sitk.Image([width, line_width], sitk.sitkUInt8) + color[1],
                sitk.Image([width, line_width], sitk.sitkUInt8) + color[2],
            ]
        )
        try:
            overlay_image[
                bbox[0] - half_line_width : bbox[0] + half_line_width + 1,  # noqa E203
                bbox[1] - half_line_width : bbox[3] + half_line_width + 1,  # noqa E203
            ] = vert
            overlay_image[
                bbox[2] - half_line_width : bbox[2] + half_line_width + 1,  # noqa E203
                bbox[1] - half_line_width : bbox[3] + half_line_width + 1,  # noqa E203
            ] = vert
            overlay_image[
                bbox[0] : bbox[2],  # noqa E203
                bbox[1] - half_line_width : bbox[1] + half_line_width + 1,  # noqa E203
            ] = horiz
            overlay_image[
                bbox[0] : bbox[2],  # noqa E203
                bbox[3] - half_line_width : bbox[3] + half_line_width + 1,  # noqa E203
            ] = horiz
        except Exception:  # Drawing outside the border of the image will cause problems
            out_of_bounds = True
            continue
    return overlay_image, out_of_bounds
