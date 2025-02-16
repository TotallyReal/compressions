# Compression functions

Trying to refresh my memory and learn more about compression techniques, I decided to write them 
down and see how it works.

My first goal was image and video compression, though text compression and general algorithms will arise naturally. In particular I got a little 
bit side tracked with Huffman codes, and ended up with a [full google colab notebook](https://colab.research.google.com/drive/14iJaLgw66eKPS27O2w4_szyeTwQTGn7U) to explain how it works. 
I didn't add the full mathematics behind it, and the interesting story of entropy, but I did partially wrote about it a long time ago, which you can [find in my homepage](https://prove-me-wrong.com/2018/04/01/how-to-measure-information-using-entropy/).

<p align="center">
  <img src="/images/english_huffman.png" width="1000"/>
</p>

Other than the general Huffman code, I already implemented the JPEG image compression, and an interesting 
new lossless compression called [The Quite OK](https://qoiformat.org/) image compression.
The next are PNG and H264, and then some more basic text compressions.

The idea right now is just to get the compressions to work, so I can understand their structure and play around with them. 

They are not implemented here to be fast (and using python certainly doesn't help). However,
if you read this and have suggestions how to make them faster, while keeping it readable, please
let me know!

I do intend in the future to create some notebooks explaining these compressions, so let me know
if you are interested in any of them.

<p align="center">
  <img src="/images/alice6.png" />
</p>

---

**My homepage**: [https://prove-me-wrong.com/](https://prove-me-wrong.com/)

**Contact**:	 [totallyRealField@gmail.com](mailto:totallyRealField@gmail.com)

**Ofir David**
