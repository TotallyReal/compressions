import random
import pytest

from codecs_funcs.huffman import HuffmanFactory, BinaryTreeEncoder, BinaryTreeRepEncoder, Node, Leaf, Branch
from codecs_funcs.encoding_functions import BitStreamListEncoder, CharToByteEncoder, ListEncoder


act_1 = """The young Duke of Athens was in a good mood as he led his bride-to-be into the garden. Quite a conquest she was, in more ways than one. She was Hippolyta, Queen of the Amazons. He had just defeated the Amazons in battle and their queen was a warrior as brave and skilled as any man he had encountered. She was beautiful and brave, and although he had won the battle, he had lost his heart. They had fallen in love and that was that.
Philostrate, a party arranger, followed them into the garden. He had come to discuss the wedding arrangements with the couple because Theseus was determined to have a big and joyous public celebration.
'Our wedding day is very close, dear Hippoplyta,' Theseus was saying as they strolled among the brightly coloured flowers. 'The new moon will rise in four days.' He stopped and looked at her. 'What a long time the old moon is taking to go down! It's making time drag in the way a long-lived stepmother or widow makes time drag for a young man who longs for his inheritance.'
Hippolyta laughed at his metaphor. 'Four days will quickly turn into four nights and we'll just dream those nights away,' she assured him. 'And soon the new moon, like a silver bow in the sky, will see the night of our celebrations.'
'Go, Philostrate,' said Theseus, 'Stir the young people of Athens up into festive mood. Get the spirit of joy moving among them. Sadness is for funerals: there's no room for it at our celebration.'
As Philostrate left on his pleasant mission, Theseus invited Hippolyta to sit down on a bench. He took her hand. 'Hippolyta,' he said, 'I pursued you as a conqueror, and won your love while doing you harm. But I'll marry you with celebration, joy and partying.'
The calm of their moment alone in the garden was interrupted by the arrival of an elderly, prominent citizen, who was arguing loudly with some attendants who wouldn't let him pass. Theseus waved his consent and Egeus strode determinedly across the lawn towards him, followed by his pretty daughter Hermia and two young men.
'Long live Theseus, our distinguished duke!' exclaimed Egeus.
'Thanks, good Egeus,' said Theseus. 'And what's the news with you?' He looked enquiringly at the three young people. Hermia was grim-faced and the two young men stared at the ground in embarrassment.
'I'm furious!' exclaimed Egeus. 'I've got a problem with my daughter Hermia.' He beckoned to one of the young men. 'Step forward, Demetrius.' He put his hand on the young man's back. 'My noble lord, this man has my consent to marry her.' He looked over his shoulder at the other young man. 'Lysander!' he barked. 'Step forward!'
Both young men now stood facing the duke, one on either side of Egeus.
'This one, my gracious duke,' wagging a finger at Lysander, 'has put some kind of spell on her. You... you, Lysander! You have given her poems, and exchanged love-tokens with my child. You have sung beneath her window by moonlight - so-called songs of so-called love! And captured her mind with bracelets of your hair, and rings and ribbons, and baubles, games, toys, knick-knacks, posies, sweets - things giddy girls are easily swayed by. You've cunningly stolen her heart and turned her obedience, which I'm entitled to, to obstinate wilfulness.' Egeus paused briefly to take a breath then rushed on. 'And, my gracious duke, if she won't consent to marry Demetrius, right here, in front of your grace, then I claim my ancient Athenian right that, as she is mine, I can do whatever I like with her. She will either marry this gentleman, Demetrius, or die according to the law in such matters.'
Theseus was a good and compassionate ruler but there were some things that were beyond his power to control. This ancient right of citizens was one of them. Egeus was only claiming his right under the law, however grim the prospect of that may be. Theseus sighed. He stood up and went to Hermia then invited her to sit beside him, which she did.
'What do you say, Hermia?' he said. 'Let me advise you, my dear young woman. Your father should be like a god to you. He's the one who gave you your beauty. Indeed, he stamped your form in wax and can reshape it or melt it as he likes.' He patted her hand. 'Demetrius is a good gentleman.'
'So is Lysander,' she said.
Theseus nodded. 'In himself he is,' he agreed. 'But in this situation, lacking your father's blessing, the other one must be regarded as the better.'
'I wish my father saw through my eyes,' she said.
Theseus shook his head. 'You have to see things his way,' he said.
Hermia cleared her throat. She glanced at Lysander then addressed the duke. 'I do beg your Grace's pardon,' she said. 'I don't know where I've found the courage to speak out, nor whether it's proper to say what I think here, but I would ask your grace to tell me what the worst thing is that could happen to me if I refuse to marry Demetrius.'
'Either to suffer death or never to have anything to do with men again,' said Theseus. 'Therefore, fair Hermia, think about what you want. Think about how young you are: examine your feelings carefully: whether, if you don't give in to your father's choice, you can endure a nun's habit: and be cooped up forever in some dark cloister, living a childless virgin all your life, singing soulless hymns to the cold, barren moon. Those who can control their passions to undertake a lifetime of virginity are thrice blessed, but the more earthly rose that gives off its perfume is happier than the one that's forced to wither on the untouched stem, growing, living and dying in single blessedness.'
'I will grow, live and die like that, my lord, rather than surrender my virginity to a husband whose unwanted authority my soul hasn't consented to accept,' she said.
Theseus stared at her. Her father was sure of his victory and the two young men waited. 'I'll give you time to think about it,' he said, at last. 'By the next new moon, the day on which my love and I seal the everlasting bond of marriage - on that day, either prepare to die for disobeying your father's demand, or else marry Demetrius, as he wishes, or take an oath of chastity and live a single, austere life forever.'
'Change your mind, sweet Hermia,' said Demetrius. 'And you, Lysander, give your ridiculous claim up to my certain right.'
Lysander laughed. 'You have her father's love, Demetrius: 'Let me have Hermia's. You marry him!'
'True, sarcastic Lysander,' said Egeus, 'he has my love. And out of that love I will give him what's mine. She is mine and I hand over all my rights in her to Demetrius.'
Lysander glanced at Hermia and she nodded slightly. He faced the duke bravely. 'My lord, I come from as good a family as he does. I'm as wealthy, I love her more than he does: my prospects are as good as, if not better than, his. And, above all, beautiful Hermia loves me. Why shouldn't I insist on my rights? Demetrius - I'll say it to his face - courted Nedar's daughter, Helena, and won her heart. And she, sweet lady, worships and idolises this flawed and unfaithful man.'
Instead of being angry Theseus listened attentively. 'I must confess that I have heard as much,' he said, 'and I was going to talk to Demetrius about it but I overlooked it because I was preoccupied with my personal affairs. But Demetrius, come with me now, and you too, Egeus, I have some advice for both of you in private. As for you, fair Hermia, steel yourself to match your desires to your father's will or else the law of Athens, which I can't change, will either condemn you to death or to a vow of lifelong chastity.' He got up. 'Come my Hippolyta.'
Hippolyta was looking at Hermes and there were tears in her eyes.
'Cheer up, my love,' said Theseus. 'Demetrius and Egeus, come along. There's something I want you to do for my wedding celebration, and also, we need to talk about those personal matters.'
When they had gone Lysander joined Hermia on the bench, where she sat, staring at nothing. He took her hand. 'Well now, my love?' he said. 'What pale cheeks? How fast the roses have faded there.'
'Probably lack of rain, which I could provide with tears.'
'Ah yes,' said Lysander. 'From all that I've read and heard, the course of true love never ran smoothly. But it was either something about the class difference or...'
'Oh what a cross to have to bear!' she exclaimed. 'Being too high-born to be allowed to fall in love with someone 'beneath' me!' She laughed in spite of herself.
'Or else our ages were badly matched...' Lysander put his thumb in his mouth and sucked it like a child.
'Oh cruel!' she exclaimed dramatically. 'Too old to be engaged to someone so young!'
'Or else your relations had something to say about it...'
'Oh hell! To have others choose one's lover!'
'Or even if everyone approved, our hope was threatened by war, death or sickness, making it as fleeting as a sound, swift as a shadow, short as a dream: as brief as lightning in the coal-black night, when it illuminates both heaven and earth in its anger,' he said. 'And before one can say 'Look!' it's swallowed by darkness again. That's how quickly bright hopes are destroyed.'
Lysander didn't seem unduly upset about these events and his light-hearted tone encouraged her. He was actually smiling! She shrugged. 'If lovers have always been crossed then it must be one of life's rules,' she said. 'We'll just have to bear it with patience if it's such a common thing: as much a part of love as thoughts, dreams and sighs, wishes, tears are - all companions of love.'
Lysander smiled broadly and took her hand. 'A good argument,' he said. 'So listen to me then, Hermia. I have a widowed aunt, an elderly lady - very rich - and she has no children. Her house is seven leagues from Athens, and she regards me as her only son. I can marry you there, darling Hermia. The harsh Athenian laws can't touch us there. If you love me, then slip out of your father's house tomorrow night and I'll wait for you in the wood a league outside town, in that place where I encountered you and Helena on that May morning.'
Ah, no wonder he was being so cheerful. He had a plan! She threw herself down on her knees and put her hand over her heart. 'Oh Lysander!' she exclaimed. 'I swear to you. By Cupid's most powerful bow: by his swiftest, gold-tipped arrow: by the... fidelity of the sacred doves of Venus...'
Lysander revolved his hand in a winding-up motion and nodded, indicating that he wanted her to continue, to swear by more things.
'Um... By whatever unites souls and makes love prosper...'
Lysander wasn't satisfied. He nodded solemnly for more.
'By the fire in which Dido destroyed herself when she saw the false Aeneas sailing away...' She tried to get up but he raised a finger for more. 'By all the vows that men have ever broken,' she said, brushing his finger aside and getting up. 'And there have been many more of those than women have ever uttered!' She sat down beside him again. 'I swear that I'll meet you in the place you mentioned.'
He kissed her hand. 'Keep your promise, my love,' he said.
There was no sign of her father, or Demetrius. They strolled to the palace gate and walked happily towards Hermes' house.
'Look, here comes Helena,' said Lysander, as that young woman swept into view, walking fast.
'God's speed, beautiful Helena!' said Hermia. 'Where are you off to?'
Helena stopped and looked suspiciously down at Hermia. She was taller, less full-figured, and, although very pretty, had nothing like the stunning looks of her friend. 'Are you calling me beautiful?' she said. 'Take that ''beautiful' back. Demetrius loves your kind of beauty. Oh lucky you! Your eyes are magnets and your voice is more pleasing than the lark's song is to a shepherd in spring, when the wheat is still green and hawthorn buds appear. I wish that looks were catching, as sickness is - I'd love to catch your looks, fair Hermia, before I go. My ear would catch your voice, my eye your eye: my tongue would catch the modulation of your voice. If I owned the world, I'd give it all, apart from Demetrius, to be you. Oh show me how to look like you and how you control Demetrius' heart!'
Hermia nodded. She had no idea. 'I frown at him and yet he still loves me,' she said.
'I wish your frowns could teach my smiles something!' exclaimed Helena.
'I curse him and he still loves me,' said Hermia.
'I wish my prayers could evoke such affection!' exclaimed Helena.
'The more I hate him the more he chases me,' said Hermia.
'The more I love him the more he hates me,' said Helena.
'It's not my fault that he's so foolish, Helena.'
'It's the fault of your beauty. I wish I had that fault!'
'Don't worry,' said Hermia. 'He won't see me again. Lysander and I are going to run away. Before I met Lysander Athens seemed like a paradise. What a wonderful man, that can turn a heaven into a hell.'
'We'll confide in you, Helena,' said Lysander. 'Tomorrow night, by moonlight, we're planning to steal through the gates of Athens.'
'And we're going to meet in the wood where you and I have often lain on the primrose beds, talking closely together. And then we're going to turn our backs on Athens and start a new life. Goodbye, dear friend. Pray for us, and good luck with Demetrius.' Hermia turned to her lover. 'Keep your promise, Lysander. We'd better stay away from each other till tomorrow night.'
She went into her house. And Lysander carried on walking, to his own house.
Helena, left on her own, walked slowly on, deep in thought. How much happier some people were than others! The whole of Athens considered her as beautiful as Hermia, but so what? Demetrius didn't think so. He refused to realise what everyone else took for granted. But just as he was mistaken in his obsession with Hermia's eyes, she was probably at fault, too, for admiring Demetrius' qualities. Love can transform things that are unpleasant and not to be admired into something beautiful and dignified. Love looks with the mind, not the eyes, and that was why Cupid is always depicted as blind. The mind of love doesn't have good judgement either: his wings and blindness suggest that he rushes into things without looking. That's why he's said to be a child: because he is so easily led. He is deceived everywhere, just as lively boys are, tricking each other all the time in games. Before Demetrius fell in love with Hermia he swore his oaths, thick as hail, to only her. And then the heat from Hermia affected the hailstorm: his love dissolved and the hail showers all melted away.
She would go and tell him about Hermia's plan to run away. Then he would also go to the wood. If he thanked her at all for the information he would do it grudgingly but it was worth it because she would follow him and at least she would be with him there and back again."""
# act_1 = act_1.replace('–','-')
# act_1 = act_1.replace('…','...')
# act_1 = act_1.replace('’','\'')
# act_1 = act_1.replace('‘','\'')


mad_tea_party ="""There was a table set out under a tree in front of the house, and the March Hare and the Hatter were having tea at it: a Dormouse was sitting between them, fast asleep, and the other two were using it as a cushion, resting their elbows on it, and talking over its head. "Very uncomfortable for the Dormouse," thought Alice; "only, as it's asleep, I suppose it doesn't mind."
The table was a large one, but the three were all crowded together at one corner of it: "No room! No room!" they cried out when they saw Alice coming. "There's plenty of room!" said Alice indignantly, and she sat down in a large arm-chair at one end of the table.
"Have some wine," the March Hare said in an encouraging tone.
Alice looked all round the table, but there was nothing on it but tea. "I don't see any wine," she remarked.
"There isn't any," said the March Hare.
"Then it wasn't very civil of you to offer it," said Alice angrily.
"It wasn't very civil of you to sit down without being invited," said the March Hare.
"I didn't know it was your table," said Alice; "it's laid for a great many more than three."
"Your hair wants cutting," said the Hatter. He had been looking at Alice for some time with great curiosity, and this was his first speech.
"You should learn not to make personal remarks," Alice said with some severity; "it's very rude."
The Hatter opened his eyes very wide on hearing this; but all he said was, "Why is a raven like a writing-desk?"
"Come, we shall have some fun now!" thought Alice. "I'm glad they've begun asking riddles.-I believe I can guess that," she added aloud.
"Do you mean that you think you can find out the answer to it?" said the March Hare.
"Exactly so," said Alice.
"Then you should say what you mean," the March Hare went on.
"I do," Alice hastily replied; "at least-at least I mean what I say-that's the same thing, you know."
"Not the same thing a bit!" said the Hatter. "You might just as well say that 'I see what I eat' is the same thing as 'I eat what I see'!"
"You might just as well say," added the March Hare, "that 'I like what I get' is the same thing as 'I get what I like'!"
"You might just as well say," added the Dormouse, who seemed to be talking in his sleep, "that 'I breathe when I sleep' is the same thing as 'I sleep when I breathe'!"
"It is the same thing with you," said the Hatter, and here the conversation dropped, and the party sat silent for a minute, while Alice thought over all she could remember about ravens and writing-desks, which wasn't much.
The Hatter was the first to break the silence. "What day of the month is it?" he said, turning to Alice: he had taken his watch out of his pocket, and was looking at it uneasily, shaking it every now and then, and holding it to his ear.
Alice considered a little, and then said "The fourth."
"Two days wrong!" sighed the Hatter. "I told you butter wouldn't suit the works!" he added looking angrily at the March Hare.
"It was the best butter," the March Hare meekly replied.
"Yes, but some crumbs must have got in as well," the Hatter grumbled: "you shouldn't have put it in with the bread-knife."
The March Hare took the watch and looked at it gloomily: then he dipped it into his cup of tea, and looked at it again: but he could think of nothing better to say than his first remark, "It was the best butter, you know."
Alice had been looking over his shoulder with some curiosity. "What a funny watch!" she remarked. "It tells the day of the month, and doesn't tell what o'clock it is!"
"Why should it?" muttered the Hatter. "Does your watch tell you what year it is?"
"Of course not," Alice replied very readily: "but that's because it stays the same year for such a long time together."
"Which is just the case with mine," said the Hatter.
Alice felt dreadfully puzzled. The Hatter's remark seemed to have no sort of meaning in it, and yet it was certainly English. "I don't quite understand you," she said, as politely as she could.
"The Dormouse is asleep again," said the Hatter, and he poured a little hot tea upon its nose.
The Dormouse shook its head impatiently, and said, without opening its eyes, "Of course, of course; just what I was going to remark myself."
"Have you guessed the riddle yet?" the Hatter said, turning to Alice again.
"No, I give it up," Alice replied: "what's the answer?"
"I haven't the slightest idea," said the Hatter.
"Nor I," said the March Hare.
Alice sighed wearily. "I think you might do something better with the time," she said, "than waste it in asking riddles that have no answers."
"If you knew Time as well as I do," said the Hatter, "you wouldn't talk about wasting it. It's him."
"I don't know what you mean," said Alice.
"Of course you don't!" the Hatter said, tossing his head contemptuously. "I dare say you never even spoke to Time!"
"Perhaps not," Alice cautiously replied: "but I know I have to beat time when I learn music."
"Ah! that accounts for it," said the Hatter. "He won't stand beating. Now, if you only kept on good terms with him, he'd do almost anything you liked with the clock. For instance, suppose it were nine o'clock in the morning, just time to begin lessons: you'd only have to whisper a hint to Time, and round goes the clock in a twinkling! Half-past one, time for dinner!"
("I only wish it was," the March Hare said to itself in a whisper.)
"That would be grand, certainly," said Alice thoughtfully: "but then-I shouldn't be hungry for it, you know."
"Not at first, perhaps," said the Hatter: "but you could keep it to half-past one as long as you liked."
"Is that the way you manage?" Alice asked.
The Hatter shook his head mournfully. "Not I!" he replied. "We quarrelled last March-just before he went mad, you know-" (pointing with his tea spoon at the March Hare,) "-it was at the great concert given by the Queen of Hearts, and I had to sing
'Twinkle, twinkle, little bat!
How I wonder what you're at!'
You know the song, perhaps?"
"I've heard something like it," said Alice.
"It goes on, you know," the Hatter continued, "in this way:-
'Up above the world you fly,
Like a tea-tray in the sky.
                    Twinkle, twinkle-'"
Here the Dormouse shook itself, and began singing in its sleep "Twinkle, twinkle, twinkle, twinkle-" and went on so long that they had to pinch it to make it stop.
"Well, I'd hardly finished the first verse," said the Hatter, "when the Queen jumped up and bawled out, 'He's murdering the time! Off with his head!'"
"How dreadfully savage!" exclaimed Alice.
"And ever since that," the Hatter went on in a mournful tone, "he won't do a thing I ask! It's always six o'clock now."
A bright idea came into Alice's head. "Is that the reason so many tea-things are put out here?" she asked.
"Yes, that's it," said the Hatter with a sigh: "it's always tea-time, and we've no time to wash the things between whiles."
"Then you keep moving round, I suppose?" said Alice.
"Exactly so," said the Hatter: "as the things get used up."
"But what happens when you come to the beginning again?" Alice ventured to ask.
"Suppose we change the subject," the March Hare interrupted, yawning. "I'm getting tired of this. I vote the young lady tells us a story."
"I'm afraid I don't know one," said Alice, rather alarmed at the proposal.
"Then the Dormouse shall!" they both cried. "Wake up, Dormouse!" And they pinched it on both sides at once.
The Dormouse slowly opened his eyes. "I wasn't asleep," he said in a hoarse, feeble voice: "I heard every word you fellows were saying."
"Tell us a story!" said the March Hare.
"Yes, please do!" pleaded Alice.
"And be quick about it," added the Hatter, "or you'll be asleep again before it's done."
"Once upon a time there were three little sisters," the Dormouse began in a great hurry; "and their names were Elsie, Lacie, and Tillie; and they lived at the bottom of a well-"
"What did they live on?" said Alice, who always took a great interest in questions of eating and drinking.
"They lived on treacle," said the Dormouse, after thinking a minute or two.
"They couldn't have done that, you know," Alice gently remarked; "they'd have been ill."
"So they were," said the Dormouse; "very ill."
Alice tried to fancy to herself what such an extraordinary way of living would be like, but it puzzled her too much, so she went on: "But why did they live at the bottom of a well?"
"Take some more tea," the March Hare said to Alice, very earnestly.
"I've had nothing yet," Alice replied in an offended tone, "so I can't take more."
"You mean you can't take less," said the Hatter: "it's very easy to take more than nothing."
"Nobody asked your opinion," said Alice.
"Who's making personal remarks now?" the Hatter asked triumphantly.
Alice did not quite know what to say to this: so she helped herself to some tea and bread-and-butter, and then turned to the Dormouse, and repeated her question. "Why did they live at the bottom of a well?"
The Dormouse again took a minute or two to think about it, and then said, "It was a treacle-well."
"There's no such thing!" Alice was beginning very angrily, but the Hatter and the March Hare went "Sh! sh!" and the Dormouse sulkily remarked, "If you can't be civil, you'd better finish the story for yourself."
"No, please go on!" Alice said very humbly; "I won't interrupt again. I dare say there may be one."
"One, indeed!" said the Dormouse indignantly. However, he consented to go on. "And so these three little sisters-they were learning to draw, you know-"
"What did they draw?" said Alice, quite forgetting her promise.
"Treacle," said the Dormouse, without considering at all this time.
"I want a clean cup," interrupted the Hatter: "let's all move one place on."
He moved on as he spoke, and the Dormouse followed him: the March Hare moved into the Dormouse's place, and Alice rather unwillingly took the place of the March Hare. The Hatter was the only one who got any advantage from the change: and Alice was a good deal worse off than before, as the March Hare had just upset the milk-jug into his plate.
Alice did not wish to offend the Dormouse again, so she began very cautiously: "But I don't understand. Where did they draw the treacle from?"
"You can draw water out of a water-well," said the Hatter; "so I should think you could draw treacle out of a treacle-well-eh, stupid?"
"But they were in the well," Alice said to the Dormouse, not choosing to notice this last remark.
"Of course they were," said the Dormouse; "-well in."
This answer so confused poor Alice, that she let the Dormouse go on for some time without interrupting it.
"They were learning to draw," the Dormouse went on, yawning and rubbing its eyes, for it was getting very sleepy; "and they drew all manner of things-everything that begins with an M-"
"Why with an M?" said Alice.
"Why not?" said the March Hare.
Alice was silent.
The Dormouse had closed its eyes by this time, and was going off into a doze; but, on being pinched by the Hatter, it woke up again with a little shriek, and went on: "-that begins with an M, such as mouse-traps, and the moon, and memory, and muchness-you know you say things are "much of a muchness"-did you ever see such a thing as a drawing of a muchness?"
"Really, now you ask me," said Alice, very much confused, "I don't think-"
"Then you shouldn't talk," said the Hatter.
This piece of rudeness was more than Alice could bear: she got up in great disgust, and walked off; the Dormouse fell asleep instantly, and neither of the others took the least notice of her going, though she looked back once or twice, half hoping that they would call after her: the last time she saw them, they were trying to put the Dormouse into the teapot.
"At any rate I'll never go there again!" said Alice as she picked her way through the wood. "It's the stupidest tea-party I ever was at in all my life!"
Just as she said this, she noticed that one of the trees had a door leading right into it. "That's very curious!" she thought. "But everything's curious today. I think I may as well go in at once." And in she went.
Once more she found herself in the long hall, and close to the little glass table. "Now, I'll manage better this time," she said to herself, and began by taking the little golden key, and unlocking the door that led into the garden. Then she went to work nibbling at the mushroom (she had kept a piece of it in her pocket) till she was about a foot high: then she walked down the little passage: and then-she found herself at last in the beautiful garden, among the bright flower-beds and the cool fountains."""

letters = list(set(mad_tea_party))
print(letters)


def test_tree_from_list():
    letters = list('abccddddeeeeeeee')
    random.shuffle(letters)
    root = HuffmanFactory.tree_from_list(letters)
    tree = BinaryTreeEncoder(root)
    assert len(tree.encode('a')) == 4
    assert len(tree.encode('b')) == 4
    assert len(tree.encode('c')) == 3
    assert len(tree.encode('d')) == 2
    assert len(tree.encode('e')) == 1


@pytest.mark.parametrize('message',['abccddddeeeeeeee', act_1, mad_tea_party])
def test_bin_tree_rep_encoder(message: str):
    tokens = [format(ord(char), '08b') for char in message]
    random.shuffle(tokens)
    root = HuffmanFactory.tree_from_list(tokens)

    encoder = BinaryTreeRepEncoder(8)
    tree_rep = encoder.encode(root)
    decoded_root = encoder.decode(tree_rep)

    def dfs(node1: Node[str], node2: Node[str]):
        assert node1.__class__ == node2.__class__, f'Nodes {node1} and {node2} must be of the same type'
        if isinstance(node1, Leaf):
            assert node1.data == node2.data, f'Leaves {node1} and {node2} must have the same data'
        else:
            dfs(node1.left, node2.left)
            dfs(node1.right, node2.right)

    dfs(root, decoded_root)

@pytest.mark.parametrize('message',['abccddddeeeeeeee', act_1, mad_tea_party])
def test_binary_tree_encoder(message: str):
    byter = ListEncoder(CharToByteEncoder())
    tokens = byter.encode(list(message))
    root = HuffmanFactory.tree_from_list(tokens)

    encoder = BitStreamListEncoder(BinaryTreeEncoder(root))

    compressed_message = encoder.encode(tokens)
    decoded = ''.join(byter.decode(encoder.decode(compressed_message)))
    assert decoded == message

def test_node_tree_to_dict():
    a_leaf = Leaf(data='a')
    b_leaf = Leaf('b')
    ab_branch = Branch(left=a_leaf,right=b_leaf)
    c_leaf = Leaf('c')
    abc_branch = Branch(left=ab_branch,right=c_leaf)

    nodes_dict, root = abc_branch.as_dict()
    assert root == '_1'
    assert nodes_dict['_1'] == ('_0','c')
    assert nodes_dict['_0'] == ('a','b')

