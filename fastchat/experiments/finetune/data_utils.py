import json
import os

wins = [1, 4, 5, 6, 7, 9, 12, 15, 18, 19, 22, 25, 27, 31, 32, 35, 39, 40, 41, 44]
rankings = [462, 440, 404, 327, 78, 345, 214, 423, 348, 424, 408, 438, 139, 350, 164, 37, 190, 328, 455, 437, 301, 262, 253, 6, 320, 26, 224, 343, 324, 371, 52, 429, 330, 119, 387, 68, 368, 342, 125, 383, 134, 318, 113, 249, 155, 303, 46, 48, 165, 24, 482, 290, 91, 120, 189, 198, 498, 494, 207, 473, 268, 497, 407, 477, 469, 105, 93, 126, 237, 420, 286, 496, 412, 88, 31, 475, 118, 416, 470, 285, 216, 276, 241, 192, 490, 393, 250, 278, 55, 114, 495, 391, 98, 305, 489, 340, 451, 332, 484, 149, 471, 166, 261, 362, 248, 479, 212, 222, 436, 463, 267, 460, 282, 14, 211, 102, 168, 252, 71, 51, 138, 411, 75, 390, 7, 140, 476, 446, 284, 410, 413, 439, 130, 25, 386, 90, 421, 493, 247, 292, 399, 41, 217, 97, 232, 441, 266, 405, 18, 259, 141, 234, 358, 481, 381, 203, 0, 146, 243, 227, 15, 472, 150, 34, 432, 220, 59, 370, 163, 372, 251, 294, 256, 144, 210, 339, 426, 143, 317, 419, 1, 442, 101, 457, 349, 16, 225, 147, 422, 193, 427, 385, 461, 194, 291, 311, 258, 299, 169, 79, 132, 109, 373, 485, 131, 254, 341, 179, 397, 274, 72, 444, 333, 230, 465, 488, 50, 43, 44, 174, 486, 112, 245, 182, 296, 242, 209, 200, 297, 400, 218, 430, 417, 275, 375, 288, 418, 491, 153, 352, 208, 233, 499, 492, 35, 40, 206, 29, 321, 99, 281, 263, 161, 336, 135, 172, 302, 458, 77, 353, 464, 2, 351, 425, 415, 323, 74, 398, 159, 435, 466, 478, 264, 452, 273, 129, 380, 191, 236, 316, 223, 21, 188, 298, 239, 315, 443, 28, 271, 152, 346, 355, 173, 367, 338, 195, 158, 22, 384, 65, 433, 365, 392, 329, 293, 87, 60, 414, 30, 363, 186, 229, 396, 357, 406, 474, 187, 326, 121, 82, 366, 142, 431, 314, 312, 468, 394, 124, 359, 295, 115, 56, 178, 277, 403, 213, 287, 459, 183, 160, 81, 448, 307, 344, 47, 369, 260, 308, 270, 89, 106, 49, 205, 306, 313, 319, 83, 64, 467, 3, 62, 42, 428, 39, 450, 255, 85, 374, 244, 111, 379, 54, 145, 123, 4, 17, 377, 157, 127, 360, 86, 12, 334, 107, 8, 447, 117, 27, 133, 175, 480, 5, 58, 61, 11, 361, 96, 116, 395, 382, 94, 23, 122, 84, 401, 181, 136, 167, 103, 197, 128, 110, 456, 33, 45, 10, 108, 310, 364, 9, 196, 13, 57, 238, 240, 309, 70, 453, 204, 378, 280, 162, 402, 32, 483, 231, 325, 487, 389, 95, 19, 449, 219, 226, 356, 347, 201, 202, 246, 80, 63, 53, 180, 283, 434, 409, 272, 335, 185, 322, 304, 69, 66, 289, 354, 265, 92, 100, 257, 171, 388, 104, 269, 221, 73, 300, 154, 38, 67, 235, 337, 151, 184, 36, 20, 176, 156, 137, 454, 331, 199, 177, 228, 76, 376, 445, 148, 279, 170, 215]
rankings = rankings[:50]

prompts_path = '/home/yusun/code/karan/data/generate/prompts.jsonl'
model_data = [
    {
    "name": "13B_beam",
    "path": "/home/yusun/code/karan/data/generate/annotations/alpaca-format/13B_beam.json",
    "output_base": "/home/yusun/code/karan/data/finetune/prompts/13B_beam/"
    },
    {
    "name": "b16",
    "path": "/home/yusun/code/karan/data/generate/annotations/alpaca-format/b16.json",
    "output_base": "/home/yusun/code/karan/data/finetune/prompts/b16/"
    },
    {
    "name": "gpt3.5",
    "path": "/home/yusun/code/karan/data/generate/annotations/alpaca-format/gpt3.5.json",
    "output_base": "/home/yusun/code/karan/data/finetune/prompts/gpt3.5/"
    },
    {
    "name": "gpt4",
    "path": "/home/yusun/code/karan/data/generate/annotations/alpaca-format/gpt4.json",
    "output_base": "/home/yusun/code/karan/data/finetune/prompts/gpt4/"
    },   
]

def format_for_finetune():
    for model in model_data:
        with open(model["path"], 'r') as file:
            data = json.load(file)

        for i, win_index in enumerate(wins):
            prompt = data[rankings[win_index]]["instruction"]
            response = data[rankings[win_index]]["output"]

            convo_data = [{
                "id": "identity_0",
                "conversations": [
                    {
                    "from": "human",
                    "value": prompt,
                    },
                    {
                    "from": "gpt",
                    "value": response
                    }
                ]
            }]
            
            os.makedirs(os.path.dirname(f"{model['output_base']}prompt{i}.json"), exist_ok=True)
            with open(f"{model['output_base']}prompt{i}.json", 'w') as file:
                json.dump(convo_data, file)

def main():
    format_for_finetune()

if __name__ == "__main__":
    main()