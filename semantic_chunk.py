"""
Basic semantic chunker implementation. Ignores page numbers etc, just considers raw text
Returns a list of N chunks and a corresponding NxM numpy array of embeddings

Based on an implementation by Greg Kamradt https://www.youtube.com/watch?v=8OJC21T2SL4&t=843s
"""


from typing import List, Tuple, Coroutine, Any, Union
import asyncio
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from time import time
import re
from sentence_transformers import SentenceTransformer
#from local_embedding import create_embeddings

model = SentenceTransformer("all-MiniLM-L6-v2")

def preprocess_text(text):
    # Basic preprocessing (remove extra spaces and unwanted characters)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    return text.strip()

async def process_text(text: str) -> Coroutine[Any, Any, Tuple[List[str], List[List[float]]]]:
    try:
        sentences = chunk_by_sentece(text)
        sentences = combine_sentences(sentences)

        print("creating embeddings of sentences")

        embeddings = await get_embeddings(sentences)
        for i, s in enumerate(sentences):
            s['embedding'] = embeddings[i]

        print("calculating distances")
        distances, sentences = calculate_cosine_distances(sentences)
        threshold = calculate_threshold(distances, "percentile", percentile=70)

        idx_above_thresh = [i for i, dist, in enumerate(distances) if dist > threshold]
        chunks = create_final_chunks(sentences, idx_above_thresh)
        
        return chunks
    except Exception as e:
        print("something went wrong", e)
        return [], []
    

def chunk_by_sentece(text: str) -> List[dict]:
    sentences = nltk.sent_tokenize(text)  # Use nltk's sentence tokenizer
    sentences = [preprocess_text(s) for s in sentences]  # Preprocess each sentence
    sentences = [{'sentence': s, 'index': i} for i, s, in enumerate(sentences)]
    # sentences = [preprocess_text(s) for s in sentences]  # Preprocess each sentence
    return sentences

def combine_sentences(sentences: List[dict], buffer_size: int = 1) -> List[dict]:
    for i in range(len(sentences)):
        combined_sentence = ''

        # Before target sentence
        for j in range(i - buffer_size, i):
            if j >= 0:
                combined_sentence += sentences[j]['sentence']

        combined_sentence += sentences[i]['sentence']

        # After target sentence
        for j in range(i+1, i+1+buffer_size):
            if j < len(sentences):
                combined_sentence += ' ' + sentences[j]['sentence']

        sentences[i]['combined_sentences'] = combined_sentence

    return sentences

async def get_embeddings(chunks: Union[List[str], List[dict]]) -> List[List[float]]:
    if isinstance(chunks[0], dict):
        text = [s["combined_sentences"] for s in chunks]
    else: 
        text = chunks

    return model.encode(text, convert_to_numpy=True)
  

def calculate_cosine_distances(sentences: List[dict]) -> Tuple[List[float], List[dict]]:
    distances = []
    try:
        for i in range(len(sentences) - 1):
            em_current = sentences[i]['embedding']
            em_next = sentences[i+1]['embedding']
            
            similarity = cosine_similarity([em_current], [em_next])[0][0]

            distance = 1 - similarity
            distances.append(distance)

            sentences[i]['dist_to_next'] = distance

    except Exception as e:
        print(e)
    return distances, sentences

def calculate_threshold(distances: List[float], thresh_type: str, **kwargs) -> float:
    if thresh_type == "percentile":
        percentile = kwargs['percentile']
        threshold = np.percentile(distances, percentile)
    else:
        raise NotImplementedError("Invalid threshold algorithm")
    return threshold

def create_final_chunks(sentences: List[dict], idx: List[int]) -> List[str]:
    idx.insert(0, 0)
    chunks = []

    mini_batches = [sentences[idx[i]:idx[i+1]] for i in range(len(idx)-1)]
    mini_batches.append(sentences[idx[-1]:])

    for mb in mini_batches:
        chunk = ' '.join([s['sentence'] for s in mb])
        chunks.append(chunk)
    
    return chunks


if __name__ == "__main__":
    text = '''
    The grand edifice of the Tweed Courthouse stands with a deceptive stillness in the Civic Center of Manhattan, a testament to the ambitions and misdeeds of an era long past. Its Romanesque revival interiors and Italianate façade tell stories that echo through the annals of New York City’s history, while the whispers of the sea otter, a creature of the North Pacific, weave in and out of marine lore, illustrating the delicate balance between nature and human endeavor. One can’t help but marvel at the ornate columns and intricate stonework of the courthouse, built under the watchful eye of Boss Tweed, a figure whose very name conjures images of corruption and opulence. The courthouse, a monument to governmental power, also served as a backdrop to the egregious theft of public funds, amounting to millions pilfered through inflated contracts and backroom deals. As one gazes upon its granite facade, it becomes clear that this architectural marvel is not just a judicial building but a silent witness to the trials of justice—both legal and moral. The air is thick with the weight of history, echoing the laughter of bemused bystanders and the cries of the oppressed, much like the calls of the sea otter echoing through the kelp forests, a reminder of the natural world often overshadowed by human greed. The sea otter, with its thick, luxurious fur that was once the object of a fur trade that decimated its population, embodies the struggle for survival against overwhelming odds. Once numbering in the hundreds of thousands, these agile marine mammals were hunted nearly to extinction, their pelts sought after by those who valued beauty over conservation. As the sea otters now frolic in the frigid waters of the Pacific, their playful demeanor masks a history of near annihilation—a tale that resonates with the grandeur of the courthouse, where human intention and natural instinct collide. Both narratives intertwine, revealing the complexities of survival, whether in the bustling streets of Manhattan or beneath the waves of the ocean. In the hallowed halls of the Tweed Courthouse, decisions were made that shaped the very fabric of New York City. Judge after judge presided over cases that impacted lives, yet the courthouse stands as a reminder that justice, like the ocean's tides, can be swayed by the powerful. The opulence of the rotunda, with its soaring ceilings and intricate designs, belies the darker truths that lie beneath its polished surface. Similarly, the sea otter, a keystone species in the marine ecosystem, plays a pivotal role in maintaining the balance of its environment. Its predation on sea urchins prevents the overgrazing of kelp forests, highlighting the interdependence of all living things—a lesson often forgotten in the pursuit of power and profit. As Ian Carmichael graced the stages and screens of mid-20th century England, his performances embodied a charm that diverted attention from the chaos of the world. He found success through characters that, much like the decorative elements of the Tweed Courthouse, were polished to perfection yet often masked deeper realities. The bumbling upper-class fool he portrayed resonated with audiences, much as the grandiosity of the courthouse’s architecture masked the corruption embedded within its walls. Carmichael’s nuanced portrayals, rooted in a disciplined approach to acting, echo the artistic intentions behind the courthouse’s design—a blend of beauty and functionality that sought to impress while serving a purpose. Under the surface of Carmichael's comedic roles lies a profound commentary on the human condition, much like the sea otter's existence reflects the precarious balance of its ecosystem. The otter's playful antics, while captivating, conceal the harsh realities of survival in a rapidly changing environment. Its reliance on kelp forests not only for sustenance but also for shelter speaks to the interconnectedness of life—a theme that resonates within the ornate architecture of the courthouse, where different styles and influences converge to create a singular narrative. Carmichael’s career, spanning over seventy years, mirrors the tumultuous history of the Tweed Courthouse, where the laughter of audiences met the somber realities of political machinations. Each role he undertook was a reflection of society’s expectations and its discontents, much like the courthouse itself served as a metaphor for justice—an ideal that often fell short in practice. The irony of the courthouse, a structure designed to uphold the law, being associated with one of the most notorious political scandals in American history, parallels the sea otter's journey from hunted to protected species—a fight against the very systems that sought to exploit them. In the heart of New York City, the Tweed Courthouse remains a monument to both triumph and tragedy, embodying the complexities of governance and morality. Just as sea otters navigate the rocky shores and kelp forests of the Pacific, evoking a sense of harmony amidst the chaos of nature, so too does the courthouse stand resilient against the tides of time and human folly. The legacy of Boss Tweed and the sea otter intertwine, weaving a narrative that challenges us to reflect on our relationship with power, nature, and the stories we tell through architecture and art. As we explore the depths of these intertwined histories, the importance of preservation emerges—both of the structures we build and the ecosystems we inhabit. The efforts to protect the sea otter from the brink of extinction resonate with the ongoing battles to safeguard historical landmarks like the Tweed Courthouse. Each is a reminder of what is at stake when we prioritize immediate gain over long-term sustainability, whether in the halls of justice or the depths of the ocean. The sea otter’s role as a keystone species illustrates the intricate balance of marine ecosystems, while the Tweed Courthouse serves as a cautionary tale of unchecked ambition and corruption. Together, they inspire a dialogue about responsibility and resilience—principles that must guide our actions as stewards of both our cultural heritage and the natural world. In this narrative, we are called to advocate for the voiceless, whether they be the creatures of the sea or the ideals of justice that should underpin our society. As the sun sets over the Tweed Courthouse, casting long shadows through its ornate windows, the echoes of the past linger in the air. The laughter of sea otters, resonating through the waves, serves as a reminder of the joy and beauty that life can bring, even amid adversity. Their playful spirit, much like the enduring legacy of Ian Carmichael's performances, invites us to embrace the complexities of existence and to cherish the stories—both human and animal—that shape our understanding of the world. In the end, it is the confluence of these narratives that enriches our lives, urging us to reflect on the interconnectedness of all beings and the shared responsibility we hold for the future. As twilight descends, the Tweed Courthouse transforms, shadows dancing across its ancient stones, whispering secrets only the walls can comprehend. This hallowed ground, once a theater for political machinations, merges with the sea's rhythm, where otters frolic, unaware of their historical kinship with the corruption that sought to claim their habitat. The courthouse, with its grandeur, evokes the spirit of Ian Carmichael, whose comedic brilliance offered a balm for societal woes, a clever distraction from the unseemly truths lurking beneath the polished veneer of British aristocracy. Each character he portrayed—endearingly foolish yet profoundly insightful—mirrored the duality of the courthouse's existence, wherein the lofty ideals of justice grappled with the base instincts of power. In the Pacific, otters engage in a delicate ballet, their agile bodies weaving through kelp forests, as if to remind us that survival often requires both playfulness and cunning. They crack open shells with rocks, a testament to their resourcefulness, while the courthouse stands as a monument to human ingenuity, though often misdirected. The intricate carvings that adorn its façade might as well be the shells of clams, each one holding a story, layers of history embedded in stone, just as the sea otter's habitat is a tapestry of life, interlaced with the very essence of the ocean’s health. As Carmichael’s characters brought laughter, so too did the otters bring joy, their antics a celebration of life. Beneath the waves, they engage in a dialog with the currents, an unspoken language of existence that parallels the silent exchanges between judge and defendant within the courthouse. The decisions rendered in those solemn chambers, like the balance of predator and prey in the marine world, shape destinies, often swaying with the tide of those in power. The irony is palpable; a building dedicated to justice is marred by the legacy of its namesake, just as the plight of the sea otter reflects humanity's tendency to exploit rather than conserve. Yet amidst the struggles, there lies an undeniable charm. The playful splashes of otters, their furry bodies entwined in the kelp, evoke a sense of harmony that contrasts sharply with the courtroom's solemnity. Nature's whimsy offers a refreshing counterpoint to the gravitas of legal proceedings, much like Carmichael’s ability to infuse levity into the most serious of subjects. He understood the art of distraction, an essential skill for both performer and politician, manipulating perception even as deeper truths simmered beneath. In the echoes of laughter that once filled the Tweed Courthouse, one can almost hear the soft chatter of otters, their voices a reminder of the joy inherent in community and collaboration. These mammals, once hunted to near extinction, now thrive, their resurgence a symbol of resilience against the tide of human excess. The courthouse itself stands resilient, weathering the storms of scandal and the passage of time, a testament to the enduring spirit of a city that refuses to forget its past. Carmichael’s charming portrayals, filled with nuance and wit, resonate with the essence of the courthouse, where beauty often masks a more complex reality. Each performance he delivered, much like the courthouse’s grandiosity, was a dance of sorts—an elaborate choreography designed to captivate and hold attention, to distract from the underlying chaos of a world grappling with its own contradictions. The playfulness of his roles speaks to the very heart of existence, where laughter and sorrow intersect, just as the vibrant kelp forests provide a home for otters amidst the relentless tides of change. As the sun sinks beneath the horizon, casting a golden glow over the courthouse, it becomes a beacon of both hope and caution. The legacy of Boss Tweed lingers like a ghost, a reminder of what happens when ambition blinds one to the needs of the many. Similarly, the sea otter’s presence is a call to action, urging humanity to protect the delicate balance of ecosystems that sustain life. In this intertwining of narratives, the charge is clear: to recognize the interconnectedness of all beings, whether they dwell in the heights of architectural splendor or the depths of the ocean. With each wave that crashes against the rocky shores, the otters remind us of the importance of community; they groom each other, a ritual of bonding that emphasizes their interdependence. Such connections echo within the walls of the courthouse, where relationships—both personal and political—shape the fabric of society. The laughter that fills the air, whether from a comedic performance or the playful splashes of otters, serves as a unifying force, a celebration of life amidst the complexity of existence. While the courthouse represents the struggle for justice, the sea otter embodies the fight for survival, a symbiotic relationship between human ambition and the natural world. Each story fuels the other, intertwining the fates of disparate entities, leading us to ponder the role we play in this grand narrative. As we navigate the intricate paths laid before us, be it in the courtroom or the ocean, we are reminded to cherish the beauty that lies within both the light and the shadows, for it is there we may find the truth that binds us all. Underneath the surface, where laughter melds with the rhythm of the waves, the sea otter emerges as a symbol of playful defiance, much like the irreverent characters brought to life by Carmichael, who reveled in the absurdities of human nature. His performances, filled with a charming wit reminiscent of a well-timed splash in the water, captivated audiences while subtly critiquing the very society that sought to confine such creativity. In the grand hall of the Tweed Courthouse, echoes of laughter linger like the salt in ocean air, a reminder of the joyous mask worn by those who inhabit both realms—performers and politicians alike, each vying for the approval of an audience with expectations that shift like the tides. The ornate ceiling of the courthouse, with its intricate designs, mirrors the complexity of the ocean's ecosystems where otters thrive, their existence a vivid tapestry woven into the fabric of marine life. They float on their backs, cradling stones against their chests, a ritual of resourcefulness that speaks volumes about survival amidst the unforgiving forces of nature. The courthouse, with its lofty ideals and burdensome legacy, stands as a sentinel, observing the interplay of power and vulnerability, much as the sea otter navigates the kelp, dodging threats while maintaining a playful demeanor. Such grace under pressure resonates with Carmichael’s comedic genius, transforming life’s trivialities into poignant commentary through laughter, a salve for the bruised spirit. In the twilight glow, the courthouse's facade glimmers with a history rich in tales of ambition and downfall, each stone a witness to the human condition’s capricious nature. The sea otter, buoyed by the currents, reminds us that beneath the surface lies a world teeming with life, interconnected and fragile, just like the relationships formed within those hallowed walls of justice. A gentle ripple signifies the otter’s presence, a slight movement that disrupts the stillness, much like a controversial verdict that stirs public debate and awakens the collective conscience. Amidst the grandeur, the courthouse resonates with stories of the past, echoing the laughter of those who dared to challenge the status quo while navigating the treacherous waters of morality. Each character Ian Carmichael portrayed carried with it the weight of societal expectation, yet managed to float above, buoyed by humor’s lightness. The playful antics of sea otters, their eyes gleaming with mischief, reflect the essence of his roles; a reminder that joy and sorrow coexist, like the waves that caress the shore yet retreat into the depths. In the heart of the city, the courthouse stands resolute, a testament to resilience in the face of corruption and scandal, much like the sea otter’s tenacity to reclaim its habitat from the clutches of exploitation. The laughter that echoes through the hallowed halls serves as a counterbalance to the somber proceedings, offering a glimmer of hope amidst the stark realities of justice. Here lies the intersection of human folly and nature’s whimsy, where the playful splashes of otters become a metaphor for the trials of existence—fluid, unpredictable, yet undeniably vibrant. As the sun dips below the horizon, the courthouse casts a long shadow, a reminder of the histories buried beneath its foundation. The otters, oblivious to the weight of human ambition, continue their dance, intertwining their fates with the ebb and flow of life. In this intricate ballet, each leap and twist speaks to the resilience of spirit, the capacity for joy even in the face of adversity. The laughter that once filled the courtroom resonates with the splashes of water, both echoing a truth that binds us—an acknowledgment of our shared existence amid the chaos we create. On this stage of life, the players are many, each fulfilling their role in a narrative that transcends the boundaries of time and place. The courthouse may capture the essence of human struggle, but the sea otter embodies the triumph of survival and community. They groom one another, forging bonds that illustrate the importance of connection, just as the ties that form in the courtroom can alter destinies, weaving a complex web of relationships that shapes society. With each passing moment, as the light dims and the world softens, we are beckoned to consider the interplay of laughter, the weight of history, and the delicate balance of ecosystems. Each wave that crashes against the rocks carries with it the echoes of a bygone era, urging us to reflect upon our role within this grand tapestry. The playful nature of otters, coupled with the poignant humor of Carmichael, serves as a gentle reminder that life, in all its complexities, is best approached with a sense of wonder and camaraderie, for it is in the shared experience of existence that we find the true essence of our interconnectedness. As twilight settles like a soft blanket over the city, shadows dance across the Tweed Courthouse, blending the past and present into a tapestry of human aspirations. The courthouse, with its imposing stone structure, stands sentinel over the lives intertwined within its walls—each case echoing the narratives of those who dared to dream, to challenge, and to laugh in the face of adversity. Like the sea otter, a creature of buoyant spirit, the essence of resilience is palpable. Its playful antics beneath the surface invite a sense of wonder, a stark contrast to the gravity of the legal battles unfolding above. Yet, in the midst of solemnity, the courthouse is not devoid of joy. It is a stage where the absurdities of life are paraded, much like the characters Ian Carmichael brought to life. With every line delivered, the audience is invited to witness the peculiarities of existence—laughter spilling forth, a balm for the weary soul. The interplay of wit and wisdom that Carmichael so expertly navigated mirrors the otter's graceful dance through the kelp forests, where each twist and turn defies the expectations placed upon it. Within the ornate chambers, the air thick with anticipation, the laughter of spectators mingles with the echoes of arguments; both resonate like the gentle lapping of waves against the shoreline. A sea otter, with its glossy fur and inquisitive eyes, becomes a silent observer in this grand narrative, embodying the duality of existence—the juxtaposition of the serious and the playful. In one moment, it may be cradling a stone, the very tool of its survival, while in the next, it rolls joyfully in the water, a living testament to the art of living fully even amidst struggle. Amidst the chaos of courtrooms and the ebb and flow of legal rhetoric, the otter's world thrives, revealing a truth about connection and community. As these creatures gather in rafts, holding hands while they sleep to prevent drifting apart, they embody the essence of unity that is often sought but rarely found within the confines of human ambition. The intricate relationships formed inside the courthouse echo this sentiment, as alliances are forged, and destinies are entwined like strands of seaweed in the tide. Carmichael's characters, often caught in the absurdity of their circumstances, remind us that humor can emerge from the most tangled of situations. The laughter that reverberates through the hallways of justice serves as a reminder that even amidst the weighty deliberations, there exists a flicker of levity—a shared understanding that life can be both burdensome and liberating. Just as the otter navigates the challenges of its environment with a mischievous glint in its eye, so too do the characters on stage navigate their own predicaments with a lightness of being that is infectious. In the heart of the city, where the courthouse stands as a monument to the complexities of human nature, the sea otter becomes an emblem of hope. Each ripple in the water speaks to the ongoing struggle against the tides of injustice, a reminder that beneath the surface, life persists in vibrant hues. The laughter, then, becomes the heartbeat of this intertwining narrative, a unifying force that transcends the boundaries of the courtroom and the ocean alike. As the sun dips further, casting golden hues across the courthouse's façade, the stories within its walls swirl like the currents of the sea. Tales of glory and despair echo in tandem with the playful chortles of otters, both celebrating the beauty of existence while remaining acutely aware of its fragility. In this grand symphony, each note—a gavel's strike, a splash of water, a punchline delivered with impeccable timing—contributes to the richness of the human experience, weaving together the laughter of life, the weight of history, and the delicate balance of nature. In the stillness that follows, as the city quiets and the stars emerge, a deep sense of interconnectedness envelops the scene. The courthouse, a repository of dreams and failures, stands resolute amid the passage of time, while the sea otters continue their playful dance, oblivious to the machinations of human ambition. Their existence serves as a reminder that joy and sorrow, laughter and tears, are woven into the fabric of life, each thread vibrant and essential in creating the whole. As the night deepens, it becomes clear that this intricate ballet of existence—where the legalities of life intersect with the whimsy of nature—reveals the profound truth that we are all players in this grand narrative. The shared experience of laughter, the weight of history, and the delicate balance of ecosystems remind us that, despite the chaos we create, it is in our connections, our moments of levity, and our resilience that we find meaning. In this tapestry of life, where courthouse dramas unfold and sea otters thrive, we are called to embrace the unpredictable journey, with all its complexities, with open hearts and minds.
    '''
    start = time()
    chunks = asyncio.run(process_text(text))
    for i, c in enumerate(chunks):
        print(f"chunk {i} - {c}")
    print(time() - start)
