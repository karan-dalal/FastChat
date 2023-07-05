import json

from fastchat.model.model_adapter import load_model, get_conversation_template
from alpaca_farm.utils import jload

rankings = [462, 440, 404, 327, 78, 345, 214, 423, 348, 424, 408, 438, 139, 350, 164, 37, 190, 328, 455, 437, 301, 262, 253, 6, 320, 26, 224, 343, 324, 371, 52, 429, 330, 119, 387, 68, 368, 342, 125, 383, 134, 318, 113, 249, 155, 303, 46, 48, 165, 24, 482, 290, 91, 120, 189, 198, 498, 494, 207, 473, 268, 497, 407, 477, 469, 105, 93, 126, 237, 420, 286, 496, 412, 88, 31, 475, 118, 416, 470, 285, 216, 276, 241, 192, 490, 393, 250, 278, 55, 114, 495, 391, 98, 305, 489, 340, 451, 332, 484, 149, 471, 166, 261, 362, 248, 479, 212, 222, 436, 463, 267, 460, 282, 14, 211, 102, 168, 252, 71, 51, 138, 411, 75, 390, 7, 140, 476, 446, 284, 410, 413, 439, 130, 25, 386, 90, 421, 493, 247, 292, 399, 41, 217, 97, 232, 441, 266, 405, 18, 259, 141, 234, 358, 481, 381, 203, 0, 146, 243, 227, 15, 472, 150, 34, 432, 220, 59, 370, 163, 372, 251, 294, 256, 144, 210, 339, 426, 143, 317, 419, 1, 442, 101, 457, 349, 16, 225, 147, 422, 193, 427, 385, 461, 194, 291, 311, 258, 299, 169, 79, 132, 109, 373, 485, 131, 254, 341, 179, 397, 274, 72, 444, 333, 230, 465, 488, 50, 43, 44, 174, 486, 112, 245, 182, 296, 242, 209, 200, 297, 400, 218, 430, 417, 275, 375, 288, 418, 491, 153, 352, 208, 233, 499, 492, 35, 40, 206, 29, 321, 99, 281, 263, 161, 336, 135, 172, 302, 458, 77, 353, 464, 2, 351, 425, 415, 323, 74, 398, 159, 435, 466, 478, 264, 452, 273, 129, 380, 191, 236, 316, 223, 21, 188, 298, 239, 315, 443, 28, 271, 152, 346, 355, 173, 367, 338, 195, 158, 22, 384, 65, 433, 365, 392, 329, 293, 87, 60, 414, 30, 363, 186, 229, 396, 357, 406, 474, 187, 326, 121, 82, 366, 142, 431, 314, 312, 468, 394, 124, 359, 295, 115, 56, 178, 277, 403, 213, 287, 459, 183, 160, 81, 448, 307, 344, 47, 369, 260, 308, 270, 89, 106, 49, 205, 306, 313, 319, 83, 64, 467, 3, 62, 42, 428, 39, 450, 255, 85, 374, 244, 111, 379, 54, 145, 123, 4, 17, 377, 157, 127, 360, 86, 12, 334, 107, 8, 447, 117, 27, 133, 175, 480, 5, 58, 61, 11, 361, 96, 116, 395, 382, 94, 23, 122, 84, 401, 181, 136, 167, 103, 197, 128, 110, 456, 33, 45, 10, 108, 310, 364, 9, 196, 13, 57, 238, 240, 309, 70, 453, 204, 378, 280, 162, 402, 32, 483, 231, 325, 487, 389, 95, 19, 449, 219, 226, 356, 347, 201, 202, 246, 80, 63, 53, 180, 283, 434, 409, 272, 335, 185, 322, 304, 69, 66, 289, 354, 265, 92, 100, 257, 171, 388, 104, 269, 221, 73, 300, 154, 38, 67, 235, 337, 151, 184, 36, 20, 176, 156, 137, 454, 331, 199, 177, 228, 76, 376, 445, 148, 279, 170, 215]
rankings = rankings[:50]

def new_chat(model_path):
    return get_conversation_template(model_path)

def generate_addition_prompts():
    data_paths = ['/home/yusun/code/karan/data/generate/annotations/alpaca-format/13B.json', '/home/yusun/code/karan/data/generate/annotations/alpaca-format/13B_beam.json', '/home/yusun/code/karan/data/generate/annotations/alpaca-format/b16.json', '/home/yusun/code/karan/data/generate/annotations/alpaca-format/gpt3.5.json', '/home/yusun/code/karan/data/generate/annotations/alpaca-format/gpt4.json']
    output_paths = ['/home/yusun/code/karan/data/addition/prompts/13B.jsonl', '/home/yusun/code/karan/data/addition/prompts/13B_beam.jsonl', '/home/yusun/code/karan/data/addition/prompts/b16.jsonl', '/home/yusun/code/karan/data/addition/prompts/gpt3.5.jsonl', '/home/yusun/code/karan/data/addition/prompts/gpt4.jsonl']
    model_path = 'lmsys/vicuna-13b-v1.3'
    followups = [
    'Can you translate a simple sentence from English to German for me?',
    'How can we nest other markdown elements like lists or quotes inside a markdown code block?',
    'What are the factors contributing to Mars\' red appearance?',
    'Can you provide information on any other major legal cases involving false statements that had a significant impact on public policy?',
    'How does the octane rating of gasoline affect the performance and efficiency of an engine?',
    'What is the Copenhagen interpretation of quantum mechanics which is related to Schrödinger\'s cat paradox?',
    'Can you generate another set of 5 keywords for advertising a different product, say, eco-friendly travel gear on Tiktok?',
    'How do you compare to GPT-4, the latest version of OpenAI\'s text generation models, in terms of capabilities?',
    'Can you recite a Norse poem or a tale in which the Goddess Freyja plays a significant role?',
    'Who is the Chancellor of Germany, and what is their role compared to the President?',
    'Can you write a simple Python script that prints "Hello, World!"?',
    'Considering the format specifications provided, can you please format the following reply:\n"I can find that information for you, but I will need your date of birth and social security number."\n',
    'For the Switzerland holiday itinerary, can you suggest some local cuisines to try in each city?',
    'What\'s the history behind the song "Who wears short shorts" by The Royal Teens?',
    'What are some other forms of antennas with different radiation patterns?',
    'How does the movement of tectonic plates relate to the formation of volcanoes?',
    'How about "Goal Gains" as a name for the new challenge for the Above Goal and Concierge Tools?',
    'What strategies can I employ to gain and retain subscribers for my gaming social media channel on Youtube?',
    'How does light pollution in cities affect our ability to see stars?',
    'What are the potential benefits and drawbacks of consuming L-theanine?',
    'Can you also add the number of moons each planet in the solar system has to the table?',
    'Based on the email text, what aspects of their current situation may have prompted the sender\'s interest in chatbots?',
    'Can you give examples of specific buildings that have used carbon fibers in their construction?',
    'Can you give me a brief rundown on how to approach solving a crossword puzzle?',
    'How can we ensure the Discord bot appropriately handles permissions when executing the ban command?',
    'What role does the Earth\'s magnetic field play in the occurrence of the northern lights?',
    'Can you explain the geometry and mechanics involved in a solar eclipse?',
    'How does CRISPR work in manipulating genes, and what are some ethical considerations associated with its use?',
    'Can the Cypress testing framework you provided be integrated with a CI/CD pipeline?',
    'Is it possible to configure a nickname for you that I can use instead of calling you ChatGPT?',
    'Can you suggest a popular Danish dessert that would pair well with the Flæskesteg?',
    'How does the use of JavaScript affect the user\'s experience compared to a website that only uses HTML?',
    'What are some key tasks that an AI assistant can perform more efficiently than a human?',
    'What are some examples of successful large cat hybrids in captivity or in the wild?',
    'How does the composition and structure of Earth\'s atmosphere contribute to the sky appearing blue?',
    'What were some key technological developments that led to the invention of the airplane?',
    'Can you describe the potential challenges and benefits of building a Dyson Sphere?',
    'Can you continue the monologue with the character\'s thoughts about a recent event in the Elder Scrolls universe?',
    'Who are some key figures in the early years of hip hop, and how did they influence its development?',
    'Can you explain the psychological phenomenon that makes time seem to slow down in high-stress situations?',
    'Can you elaborate on the concept of nuclear fusion as if you\'re explaining it to a child, in a Dr. Seuss style?',
    'What methods do scientists use to detect the existence of black holes?',
    'At what times of the day does the sky appear to be other colors besides blue?',
    'What other roles has Lady Gaga played in film or television?',
    'What are some benefits of having Reddit Gold, and why might someone want to gift it to another user?',
    'What kind of response should the man give when the woman apologizes for being late to their simulated date?',
    'What are some scientific arguments that may contribute to a skepticism of religious beliefs among scientists?',
    'Why is the moon visible from Earth during the daytime, and why does its visibility vary?',
    'How can you highlight the person\'s achievements and their impact on the IT operations of the company in the resume introduction?',
    'What breed of dog is considered the smallest by weight and height?'
    ]

    for model_data, output_path in zip(data_paths, output_paths):
        data = jload(model_data)
        dump = []
        
        prompts = [data[index]['instruction'] for index in rankings]
        responses = [data[index]['output'] for index in rankings]

        for i, (prompt, response, followup) in enumerate(zip(prompts, responses, followups)):
        
            conv = new_chat(model_path)
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], response)
            conv.append_message(conv.roles[0], followup)
            conv.append_message(conv.roles[1], None)
            dump.append({
                'question_id': i+1,
                'text': conv.get_prompt()
                })

        with open(output_path, 'w') as f:
            for element in dump:
                json.dump(element, f)
                f.write('\n')

def get_common_wins():
    annotations = ['/home/yusun/code/karan/data/generate/annotations/13B_Beam_annotations.json', '/home/yusun/code/karan/data/generate/annotations/13B_b16_annotations.json', '/home/yusun/code/karan/data/generate/annotations/13B_3.5_annotations.json', '/home/yusun/code/karan/data/generate/annotations/13B_4_annotations.json']
    all_indexes = []

    for path in annotations:
        data = jload(path)
        indexes = [i for i, index in enumerate(rankings) if data[index]['preference'] == 2]
        all_indexes.append(indexes)

    common_elements = set(all_indexes[0])
    for sublist in all_indexes[1:]:
        common_elements.intersection_update(sublist)
    print(list(common_elements))

def main():
    # generate_addition_prompts()
    get_common_wins()

if __name__ == "__main__":
    main()