import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import argparse

from accelerate.utils import set_seed

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from libs.engine import merge_and_update_config
from libs.utils.argparse import accelerate_parser, base_data_parser
from pipelines.painter.diffsketchedit_pipeline import DiffSketchEditPipeline


class PromptInfo:
    def __init__(self, prompts, token_ind, changing_region_words, reweight_word=None, reweight_weight=None):
        self.prompts = prompts
        self.token_ind = token_ind
        self.changing_region_words = changing_region_words
        self.reweight_word = reweight_word
        self.reweight_weight = reweight_weight


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="vary style and content painterly rendering",
        parents=[accelerate_parser(), base_data_parser()]
    )
    # config
    parser.add_argument("-c", "--config",
                        type=str,
                        default="diffsketchedit.yaml",
                        help="YAML/YML file for configuration.")

    parser.add_argument("-style", "--style_file",
                        default="", type=str,
                        help="the path of style img place.")

    # result path
    parser.add_argument("-respath", "--results_path",
                        type=str, default="./workdir",
                        help="If it is None, it is automatically generated.")
    parser.add_argument("-npt", "--negative_prompt", default="", type=str)

    parser.add_argument("--sd_image_only", default=0, type=int,
                        help="1 for generating the SD images only; 0 for generating the subsequent vector sketches.")

    parser.add_argument("--vector_local_edit", default=1, type=int)
    parser.add_argument("--vector_local_edit_bin_threshold_replace", default=0.3, type=float)
    parser.add_argument("--vector_local_edit_bin_threshold_refine", default=0.3, type=float)
    parser.add_argument("--vector_local_edit_bin_threshold_reweight", default=0.3, type=float)
    parser.add_argument("--vector_local_edit_attn_res", default=16, choices=[16, 32, 64], type=int)

    # DiffSVG
    parser.add_argument("--print_timing", "-timing", action="store_true",
                        help="set print svg rendering timing.")
    # diffuser
    parser.add_argument("--download", default=0, type=int,
                        help="download models from huggingface automatically.")
    parser.add_argument("--force_download", "-download", action="store_true",
                        help="force the models to be downloaded from huggingface.")
    parser.add_argument("--resume_download", "-dpm_resume", action="store_true",
                        help="download the models again from the breakpoint.")
    # rendering quantity
    # like: python main.py -rdbz -srange 100 200
    parser.add_argument("--render_batch", "-rdbz", action="store_true")
    parser.add_argument("-srange", "--seed_range",
                        required=False, nargs='+',
                        help="Sampling quantity.")
    # visual rendering process
    parser.add_argument("-mv", "--make_video", action="store_true",
                        help="make a video of the rendering process.")
    parser.add_argument("-frame_freq", "--video_frame_freq",
                        default=1, type=int,
                        help="video frame control.")
    args = parser.parse_args()

    args = merge_and_update_config(args)

    ############################### main parameters ###############################

    seeds_list = [25760]
    # seeds_list = [random.randint(1, 65536) for _ in range(100)]
    args.edit_type = "replace"  # ["replace", "refine", "reweight"]
    prompt_infos = [
        ## "replace" examples
        PromptInfo(prompts=["A painting of a squirrel eating a burger",
                            "A painting of a rabbit eating a burger",
                            "A painting of a rabbit eating a pumpkin",
                            "A painting of a owl eating a pumpkin"],
                   token_ind=5,
                   changing_region_words=[["", ""], ["squirrel", "rabbit"], ["burger", "pumpkin"], ["rabbit", "owl"]]),

        # PromptInfo(prompts=["A boy wearing a cap",
        #                     "A boy wearing a beanie"],
        #            token_ind=2,
        #            changing_region_words=[["", ""], ["cap", "beanie"]]),

        # PromptInfo(prompts=["A desk near the bookshelf",
        #                     "A chair near the bookshelf"],
        #            token_ind=2,
        #            changing_region_words=[["", ""], ["desk", "chair"]]),

        ## "refine" examples
        # PromptInfo(prompts=["An evening dress",
        #                     "An evening dress with sleeves",
        #                     "An evening dress with sleeves and a belt"],
        #            token_ind=3,
        #            changing_region_words=[["", ""], ["", "sleeves"], ["", "belt"]]),

        ## "reweight" examples
        # PromptInfo(prompts=["An emoji face with moustache and smile"] * 3,
        #            token_ind=3,
        #            changing_region_words=[["", ""], ["moustache", "moustache"], ["smile", "smile"]],
        #            reweight_word=["moustache", "smile"],
        #            reweight_weight=[-1.0, 3.0]),

        # PromptInfo(prompts=["A photo of a birthday cake with candles"] * 2,
        #            token_ind=6,
        #            changing_region_words=[["", ""], ["candles", "candles"]],
        #            reweight_word=["candles"],
        #            reweight_weight=[-5.0])
    ]

    ############################### main parameters (end) ###############################

    args.batch_size = 1  # rendering one SVG at a time
    pipe = DiffSketchEditPipeline(args)

    for seed in seeds_list:
        for prompt_info in prompt_infos:
            run_stages = len(prompt_info.prompts)
            for run_stage in range(run_stages):
                args.run_stage = run_stage
                set_seed(seed)
                pipe.update_info(seed, prompt_info.token_ind, prompt_info.prompts[0])
                pipe.painterly_rendering(prompt_info.prompts,
                                         prompt_info.token_ind, prompt_info.changing_region_words,
                                         reweight_word=prompt_info.reweight_word, reweight_weight=prompt_info.reweight_weight)
                pipe.close(msg="painterly rendering complete.")
