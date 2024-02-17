import torch


def get_answers(answer):
    if answer['gender'] == "male":
        temp = "man is wearing "
    elif answer['gender'] == "female":
        temp = "woman is wearing "
    else:
        temp = "person is wearing "
    if answer['gender'] == "male":
        neg_temp = "The man is with "
    elif answer['gender'] == "female":
        neg_temp = "The woman is not with "
    else:
        neg_temp = "The person is not wearing "
    top = answer['top_color'] + " " + answer['tops'] + "."
    lower = answer['low_color'] + " " + answer['pants'] + "."
    shoes = answer['shoes_color'] + " " + answer['shoes'] + "."
    res = [temp+top, temp+lower, temp+shoes]
    if answer["bags"] == 'yes':
        res.append(temp+"bag")
    else:
        res.append(neg_temp+"bag")
    if answer["glasses"] == 'yes':
        res.append(temp+"glasses")
    else:
        res.append(neg_temp+"glasses")
    # res.append(answer['outlook'])

    return res

def generate_caption_vqa(image, model_vqa):
    # split_captions = origin_caption.split(' ')
    num_image = image.shape[0]
    questions = {
    "gender" : f"What is the gender of this person?",
    "outlook": f"What dose the person look like?",
    "tops": f"What upper garment is this person wearing?",
    "top_color": f"What color of upper garment is this person wearing?",
    "pants": f"What lower garment is this person wearing",
    "low_color": f"What color of lower garment is this person wearing?",
    "shoes": f"What shoes is this person wearing?",
    "shoes_color": f"What color of shoes is this person wearing?",
    # "posture": f"What is the posture of this person?",
    "bags": f"Is this person carring bags?",
    "glasses": f"Is this person wearing glasses?",
    }
    answer = {}
    for k, v in questions.items():
        temp_answer = model_vqa.generate(image, v, train=False, inference='generate', num_frames=num_image)
        answer[k] = temp_answer[0]
    res = get_answers(answer)
    return res

