"""utils used in this module"""

import numpy as np
import pickle
import inspect
import logging
import os
import pathlib
import shutil
import json
import warnings
import joblib
import time
import pandas as pd
import torch.multiprocessing as mp

from datetime import timedelta
from pathlib import Path
from google.cloud import documentai
from concurrent.futures import ProcessPoolExecutor, as_completed

from termcolor import colored
from tqdm import tqdm


def read_dat(path: Path, convert_mpp=True, cohort="TCGA"):
    """
    Expects a dat file with dictionary with following schema:
    source-image-resolution - {'objective_power':.., 'mpp': .., 'dimensions': [x, y]} (metadata from original slide)
    element-resolution - {'units': .., 'resolution': ..} (inference parameters used by model)
    elements - n x {'box': .., 'centroid': .., 'contour': .., 'prob': .., 'type': ..} (information of each nuclei instance)
    """
    data = joblib.load(path)
    source_meta = data["source-image-resolution"]
    infer_meta = data["element-resolution"]
    elements = data["elements"]
    mpp_ratio = infer_meta["resolution"] / source_meta["mpp"]

    if cohort.upper() == "TCGA":
        case_id = Path(path).stem.split(".")[0][:-11]
    elif cohort.upper() == "BC2001":
        slide_stem = Path(path).stem
        slide_mapping = joblib.load("metadata/bc2001_slide_mapping.dat")
        case_id = slide_mapping[slide_stem]
    elif cohort.upper() == "UHCW":
        case_id = Path(path).stem.split("_")[0]

    elements = [pd.Series(elements[key]) for key in elements]
    elements = pd.DataFrame(elements)

    if convert_mpp:
        # convert elements to slide level
        elements["box"] = elements["box"] * mpp_ratio[0]
        elements["contour"] = elements["contour"] * mpp_ratio[0]
        elements["centroid"] = elements["centroid"] * mpp_ratio[0]
        return elements, source_meta, infer_meta, mpp_ratio[0], case_id
    else:
        return elements, source_meta, infer_meta, case_id


def check_unit_conversion_integrity(
    input_unit, output_unit, baseline_mpp, baseline_power
):
    """Checks integrity of units before unit conversion.

    Args:
        input_unit (str):
            input units
        output_unit (str):
            output units
        baseline_mpp:
            baseline microns per pixel (mpp)
        baseline_power:
            baseline magnification level.

    Raises:
        ValueError:
            If the checks on unit conversion fails.

    """
    if input_unit not in {"mpp", "power", "baseline"}:
        raise ValueError(
            "Invalid input_unit: argument accepts only one of the following "
            " options: `'mpp'`, `'power'`, `'baseline'`."
        )
    if output_unit not in {"mpp", "power", "baseline", None}:
        raise ValueError(
            "Invalid output_unit: argument accepts only one of the following"
            " options: `'mpp'`, `'power'`, `'baseline'`, or None (to return"
            " all units)."
        )
    if baseline_mpp is None and input_unit == "mpp":
        raise ValueError(
            "Missing 'mpp': `input_unit` has been set to 'mpp' while there "
            "is no information about 'mpp' in WSI meta data."
        )
    if baseline_power is None and input_unit == "power":
        raise ValueError(
            "Missing 'objective_power': `input_unit` has been set to 'power' while "
            "there is no information about 'objective_power' in WSI meta data."
        )


def prepare_output_dict(input_unit, input_res, baseline_mpp, baseline_power):
    # calculate the output_res based on input_unit and resolution
    output_dict = {
        "mpp": None,
        "power": None,
        "baseline": None,
    }
    if input_unit == "mpp":
        if isinstance(input_res, (list, tuple, np.ndarray)):
            output_dict["mpp"] = np.array(input_res)
        else:
            output_dict["mpp"] = np.array([input_res, input_res])
        output_dict["baseline"] = baseline_mpp[0] / output_dict["mpp"][0]
        if baseline_power is not None:
            output_dict["power"] = output_dict["baseline"] * baseline_power
        return output_dict
    if input_unit == "power":
        output_dict["baseline"] = input_res / baseline_power
        output_dict["power"] = input_res
    elif input_unit == "level":
        raise ValueError("Can't deal with levels atm")
        # level_scales = relative_level_scales(input_res, input_unit)
        # output_dict["baseline"] = level_scales[0]
        # if baseline_power is not None:
        #     output_dict["power"] = output_dict["baseline"] * baseline_power
    else:  # input_unit == 'baseline'
        output_dict["baseline"] = input_res
        if baseline_power is not None:
            output_dict["power"] = baseline_power * output_dict["baseline"]

    if baseline_mpp is not None:
        output_dict["mpp"] = baseline_mpp / output_dict["baseline"]

    return output_dict


def convert_resolution_units(source_meta, input_res, input_unit, output_unit=None):
    baseline_mpp = source_meta["mpp"][0]
    baseline_power = source_meta["objective_power"]

    check_unit_conversion_integrity(
        input_unit, output_unit, baseline_mpp, baseline_power
    )

    output_dict = prepare_output_dict(
        input_unit, input_res, baseline_mpp, baseline_power
    )
    out_res = output_dict[output_unit] if output_unit is not None else output_dict
    if out_res is None:
        warnings.warn(
            "Although unit conversion from input_unit has been done, the requested "
            "output_unit is returned as None. Probably due to missing 'mpp' or "
            "'objective_power' in slide's meta data.",
            UserWarning,
        )
    return out_res


def get_bounding_box(img: np.ndarray) -> np.ndarray:
    """Get the bounding box coordinates of a binary input- assumes a single object.

    Args:
        img: input binary image.

    Returns:
        bounding box coordinates

    """
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return np.array([rmin, rmax, cmin, cmax])


def rm_n_mkdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)
    return


def rmdir(dir_path):
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    return


def mkdir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return


def recur_find_ext(root_dir, ext_list):
    """
    recursively find all files in directories end with the `ext`
    such as `ext='.png'`

    return list is alrd sorted
    """
    assert isinstance(ext_list, list)
    file_path_list = []
    for cur_path, dir_list, file_list in os.walk(root_dir):
        for file_name in file_list:
            file_ext = pathlib.Path(file_name).suffix
            if file_ext in ext_list:
                full_path = os.path.join(cur_path, file_name)
                file_path_list.append(full_path)
    file_path_list.sort()
    return file_path_list


def log_debug(msg):
    (
        frame,
        filename,
        line_number,
        function_name,
        lines,
        index,
    ) = inspect.getouterframes(inspect.currentframe())[1]
    line = lines[0]

    indentation_level = line.find(line.lstrip())
    logging.debug("{i} {m}".format(i="." * indentation_level, m=msg))


def log_info(msg):
    (
        frame,
        filename,
        line_number,
        function_name,
        lines,
        index,
    ) = inspect.getouterframes(inspect.currentframe())[1]

    line = lines[0]
    indentation_level = line.find(line.lstrip())
    logging.info("{i} {m}".format(i="." * indentation_level, m=msg))


def wrap_func(idx, func, *args):
    try:
        return idx, func(*args)
    except Exception as e:
        return e, idx, None


def multiproc_dispatcher(
    data_list, num_workers=0, show_pbar=True, crash_on_exception=False
):
    """
    data_list is alist of [[func, arg1, arg2, etc.]]
    Resutls are alway sorted according to source position
    """
    if num_workers > 0:
        proc_pool = ProcessPoolExecutor(num_workers)

    result_list = []
    future_list = []

    if show_pbar:
        pbar = tqdm(total=len(data_list), ascii=True, position=0)
    for run_idx, dat in enumerate(data_list):
        func = dat[0]
        args = dat[1:]
        if num_workers > 0:
            future = proc_pool.submit(wrap_func, run_idx, func, *args)
            future_list.append(future)
        else:
            # ! assume 1st return is alwasy run_id
            result = wrap_func(run_idx, func, *args)
            if len(result) == 3 and crash_on_exception:
                raise result[0]
            elif len(result) == 3:
                result = result[1:]
            result_list.append(result)
            if show_pbar:
                pbar.update()
    if num_workers > 0:
        for future in as_completed(future_list):
            if future.exception() is not None:
                if crash_on_exception:
                    raise future.exception()
                logging.info(future.exception())
            else:
                result = future.result()
                if len(result) == 3 and crash_on_exception:
                    raise result[0]
                elif len(result) == 3:
                    result = result[1:]
                result_list.append(result)
            if show_pbar:
                pbar.update()
        proc_pool.shutdown()
    if show_pbar:
        pbar.close()

    result_list = sorted(result_list, key=lambda k: k[0])
    result_list = [v[1:] for v in result_list]
    return result_list


def worker_func(run_idx, func, crash_on_exception, *args):
    result = func(*args)
    if len(result) == 3 and crash_on_exception:
        raise result[0]
    elif len(result) == 3:
        result = result[1:]
    return run_idx, result


def multiproc_dispatcher_torch(
    data_list, num_workers=0, show_pbar=True, crash_on_exception=False
):
    """
    data_list is a list of [[func, arg1, arg2, etc.]]
    Results are always sorted according to source position
    """
    result_list = []

    if show_pbar:
        pbar = tqdm(total=len(data_list), ascii=True, position=0)

    if num_workers > 0:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(num_workers)
        for run_idx, dat in enumerate(data_list):
            func = dat[0]
            args = dat[1:]
            result = pool.apply_async(
                worker_func, (run_idx, func, crash_on_exception, *args)
            )
            result_list.append(result)
            if show_pbar:
                pbar.update()
        pool.close()
        pool.join()
        result_list = [result.get() for result in result_list]
    else:
        for run_idx, dat in enumerate(data_list):
            func = dat[0]
            args = dat[1:]
            result = worker_func(run_idx, func, crash_on_exception, *args)
            result_list.append(result)
            if show_pbar:
                pbar.update()

    if show_pbar:
        pbar.close()

    result_list = sorted(result_list, key=lambda k: k[0])
    result_list = [v[1] for v in result_list]
    return result_list


def convert_pytorch_checkpoint(net_state_dict):
    variable_name_list = list(net_state_dict.keys())
    is_in_parallel_mode = all(v.split(".")[0] == "module" for v in variable_name_list)
    if is_in_parallel_mode:
        colored_word = colored("WARNING", color="red", attrs=["bold"])
        print(
            (
                "%s: Detect checkpoint saved in data-parallel mode."
                " Converting saved model to single GPU mode." % colored_word
            ).rjust(80)
        )
        net_state_dict = {
            ".".join(k.split(".")[1:]): v for k, v in net_state_dict.items()
        }
    return net_state_dict


def difference_filename(listA, listB):
    """Return paths in A that dont have filename in B."""
    name_listB = [pathlib.Path(v).stem for v in listB]
    name_listB = list(set(name_listB))
    name_listA = [pathlib.Path(v).stem for v in listA]
    sel_idx_list = []
    for idx, name in enumerate(name_listA):
        try:
            name_listB.index(name)
        except ValueError:
            sel_idx_list.append(idx)
    if len(sel_idx_list) == 0:
        return []
    sublistA = np.array(listA)[np.array(sel_idx_list)]
    return sublistA.tolist()


def intersection_filename(listA, listB):
    """Return paths with file name exist in both A and B."""
    name_listA = [pathlib.Path(v).stem for v in listA]
    name_listB = [pathlib.Path(v).stem for v in listB]
    union_name_list = list(set(name_listA).intersection(set(name_listB)))
    union_name_list.sort()

    sel_idx_list = []
    for _, name in enumerate(union_name_list):
        try:
            sel_idx_list.append(name_listA.index(name))
        except ValueError:
            pass
    if len(sel_idx_list) == 0:
        return [], []
    sublistA = np.array(listA)[np.array(sel_idx_list)]

    sel_idx_list = []
    for _, name in enumerate(union_name_list):
        try:
            sel_idx_list.append(name_listB.index(name))
        except ValueError:
            pass
    sublistB = np.array(listB)[np.array(sel_idx_list)]

    return sublistA.tolist(), sublistB.tolist()


def load_json(path):
    with open(path, "r") as fptr:
        return json.load(fptr)


def set_logger(path):
    logging.basicConfig(level=logging.INFO)
    # * reset logger handler
    log_formatter = logging.Formatter(
        "|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
    )
    log = logging.getLogger()  # root logger
    for hdlr in log.handlers[:]:  # remove all old handlers
        log.removeHandler(hdlr)
    new_hdlr_list = [logging.FileHandler(path), logging.StreamHandler()]
    for hdlr in new_hdlr_list:
        hdlr.setFormatter(log_formatter)
        log.addHandler(hdlr)


def timeit(start):
    end = time.perf_counter()
    total = str(timedelta(seconds=end - start))
    return total


def contains_tissue(im, color_threshold=200, percentage_threshold=0.6):
    """
    If more than 60% pixels have mean value above 200 ignore those.
    """
    return (
        np.mean(np.mean(im, axis=2).flatten() > color_threshold) < percentage_threshold
    )


def find_if_multiple_slides(file_list, case_id):
    # file_list = [Path(x).stem for x in file_list]
    case_slides = [x for x in file_list if Path(x).stem.split("-01Z")[0] == case_id]
    if len(case_slides) > 1:
        return True, case_slides
    else:
        return False, case_slides


def parse_blocks(data):
    n_blocks = len(data["blocks"].keys())
    page_text = ""
    for block_idx in range(n_blocks):
        block_text = data["blocks"][f"block_{block_idx}"]["text"]
        page_text += block_text
    return page_text


def parse_pickle(data):
    # extract pages from pickle file
    n_pages = data["number_of_page"]

    page_dict = {}
    for i in range(1, n_pages + 1):
        page_dict[f"page_{i}"] = parse_blocks(data[f"page_{i}"])
    return page_dict


def parse_pickle_full(doc):
    data = clean_pickle(doc)

    # extract pages from pickle file
    n_pages = data["number_of_page"]

    page_dict = {}
    for i in range(1, n_pages + 1):
        page_dict[f"page_{i}"] = parse_blocks(data[f"page_{i}"])
    return page_dict


def get_page_from_pickle(pickle_path, page_num=1):
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    page_dict = {"page_1": parse_blocks(data[f"page_{page_num}"])}
    return page_dict


def layout_to_text(layout: documentai.Document.Page.Layout, text: str) -> str:
    """
    Document AI identifies text in different parts of the document by their
    offsets in the entirety of the document"s text. This function converts
    offsets to a string.
    """
    # If a text segment spans several lines, it will
    # be stored in different text segments.
    return "".join(
        text[int(segment.start_index) : int(segment.end_index)]
        for segment in layout.text_anchor.text_segments
    )


def clean_pickle(document):
    text = document.text
    out_json = {"text": text}
    out_json["number_of_page"] = len(document.pages)

    for page in document.pages:
        page_out = {"page_number": page.page_number}
        page_out["width"] = page.dimension.width
        page_out["height"] = page.dimension.height

        page_out_blocks = {}
        for idx, block in enumerate(page.blocks):
            page_out_blocks[f"block_{idx}"] = {
                "text": layout_to_text(block.layout, text),
                "layout": block.layout,
            }

        page_out["blocks"] = page_out_blocks

        page_out_paragraphs = {}
        for idx, paragraph in enumerate(page.paragraphs):
            page_out_paragraphs[f"paragraph_{idx}"] = {
                "text": layout_to_text(paragraph.layout, text),
                "layout": paragraph.layout,
            }
        page_out["paragraphs"] = page_out_paragraphs

        page_out_lines = {}

        for idx, line in enumerate(page.lines):
            page_out_paragraphs[f"line_{idx}"] = {
                "text": layout_to_text(line.layout, text),
                "layout": paragraph.layout,
            }
        page_out["lines"] = page_out_lines

        page_out_tokens = {}
        for idx, token in enumerate(page.tokens):
            page_out_paragraphs[f"token_{idx}"] = {
                "text": layout_to_text(token.layout, text),
                "layout": paragraph.layout,
            }
        page_out["tokens"] = page_out_tokens
        out_json[f"page_{page.page_number}"] = page_out
    return out_json
