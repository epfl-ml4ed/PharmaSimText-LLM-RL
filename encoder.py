from sentence_transformers import SentenceTransformer
import json
import os
import numpy as np
import fasttext
import fasttext.util
import shutil
# fasttext.util.download_model('en', if_exists='ignore')
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer

class Encoder():
    def __init__(self, encoder_type="fasttext", emb_dim=300, memory_size=1000, save_memory=0.1, memory_path='../encoder_memory/memory.json'):
        self.memory_path = memory_path
        self.encoder_type = encoder_type
        if encoder_type == "fasttext":
            if not os.path.exists("./lms/cc.en.300.bin"):
                fasttext.util.download_model('en')
                shutil.move("./cc.en.300.bin", "./lms/cc.en.300.bin")
            self.model = fasttext.load_model(
                "./lms/cc.en.300.bin"
            )
            if emb_dim < 300:
                print("reducing fasttext model...")
                fasttext.util.reduce_model(self.model, emb_dim)

        elif encoder_type == "bert" or encoder_type == "medbert":
            self.model = SentenceTransformer(model_name_or_path='all-MiniLM-L6-v2' if encoder_type == "bert" else 'pritamdeka/S-PubMedBert-MS-MARCO')
        else:
            raise NotImplementedError

        self.memory = {}
        if os.path.exists(memory_path):
            self.memory = json.load(open(memory_path, 'r'))
        self.memory_size = memory_size
        self.save_memory = save_memory

    def encode(self, sentences):
        encoded = []
        for s in sentences:
            if s not in self.memory.keys():
                x = self.model.get_sentence_vector(s) if self.encoder_type == "fasttext" else self.model.encode([s])
                x = x.reshape(1, -1)
                if len(self.memory)+1 > self.memory_size:
                    self.memory.popitem()
                self.memory[s] = x.tolist()
                if len(self.memory) % (round(self.memory_size*self.save_memory)) == 0:
                    json.dump(self.memory, open(self.memory_path, 'w'))
            encoded.append(np.array(self.memory[s]))
        return encoded


def main():
    e = Encoder(encoder_type="medbert", emb_dim=300, memory_size=10, save_memory=0.1, memory_path='../encoder_memory/medbert.json')
    embs = e.encode([
        "The mysterious old book sat on the dusty shelf, its pages filled with forgotten tales.",
        "With a sudden gust of wind, the leaves danced in a chaotic frenzy, painting the autumn sky with shades of gold and crimson.",
        "As the first rays of sunlight peeked over the horizon, the sleepy town slowly awakened to a new day.",
        "In the heart of the bustling city, a street performer captivated the crowd with mesmerizing melodies on a weathered violin.",
        "The aroma of freshly baked bread wafted through the air, enticing passersby to step into the quaint bakery on the corner.",
        "Beneath the twinkling stars, a lone wolf howled, its mournful cry echoing through the silent night.",
        "Lost in thought, she absentmindedly traced the rim of her coffee cup with her fingertips, the steam curling upwards in delicate wisps.",
        "The antique clock on the mantelpiece ticked away the hours, a faithful guardian of time in the quiet room.",
        "Amidst the vibrant market stalls, a street vendor skillfully juggled colorful fruits, drawing smiles from amused onlookers.",
        "A rusted key, hidden for decades, was discovered in the dusty attic, unlocking memories of a bygone era.",
        "The old lighthouse stood tall against the stormy sea, its beacon guiding ships safely through the turbulent waters.",
        "With a mischievous grin, the child presented a hand-picked bouquet of wildflowers, a simple gesture that warmed the heart.",
        "The scent of rain lingered in the air as thunder rumbled in the distance, signaling an approaching summer storm.",
        "A tattered map led the adventurous explorer through dense jungles, across roaring rivers, and finally to a hidden treasure trove.",
        "As the full moon bathed the landscape in silver light, the nocturnal creatures emerged from their hiding places, beginning their nightly rituals.",
        "The old oak tree, with branches reaching towards the sky, whispered tales of centuries gone by to anyone who cared to listen.",
        "In the quiet library, the turning of pages and the occasional cough created a symphony of knowledge and contemplation.",
        "Wandering through a field of sunflowers, she felt the warmth of the sun on her face and the gentle embrace of a soft breeze.",
        "A single candle flickered in the darkness, casting eerie shadows on the walls and revealing glimpses of a forgotten room.",
        "The laughter of children echoed through the playground, a joyful melody that echoed the innocence of youth."
    ])
    print(embs)
    print(embs[0].shape)


if __name__ == '__main__':
    main()
