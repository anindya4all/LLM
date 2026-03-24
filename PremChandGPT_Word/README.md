
# premchandGPT

![nanoGPT](assets/premchandgpt.png)


The simplest, fastest repository for training/finetuning medium-sized GPTs. It is a rewrite of [minGPT](https://github.com/karpathy/minGPT) that prioritizes teeth over education. Still under active development, but currently the file `train.py` reproduces GPT-2 (124M) on OpenWebText, running on a single 8XA100 40GB node in about 4 days of training. The code itself is plain and readable: `train.py` is a ~300-line boilerplate training loop and `model.py` a ~300-line GPT model definition, which can optionally load the GPT-2 weights from OpenAI. That's it.

![repro124m](assets/gpt2_124M_loss.png)

Because the code is so simple, it is very easy to hack to your needs, train new models from scratch, or finetune pretrained checkpoints (e.g. biggest one currently available as a starting point would be the GPT-2 1.3B model from OpenAI).

## install

```
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3
-  `transformers` for huggingface transformers <3 (to load GPT-2 checkpoints)
-  `datasets` for huggingface datasets <3 (if you want to download + preprocess OpenWebText)
-  `tiktoken` for OpenAI's fast BPE code <3
-  `wandb` for optional logging <3
-  `tqdm` for progress bars <3

## quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a character-level GPT on the works of premchand. First, we download it as a single (1MB) file and turn it from raw text into one large stream of integers:

```sh
python data/premchand_char/prepare.py
```

This creates a `train.bin` and `val.bin` in that data directory. Now it is time to train your GPT. The size of it very much depends on the computational resources of your system:

**I have a GPU**. Great, we can quickly train a baby GPT with the settings provided in the [config/train_premchand_char.py](config/train_premchand_char.py) config file:

```sh
python train.py config/train_premchand_char.py
```

If you peek inside it, you'll see that we're training a GPT with a context size of up to 256 characters, 384 feature channels, and it is a 6-layer Transformer with 6 heads in each layer. On one A100 GPU this training run takes about 3 minutes and the best validation loss is 1.4697. Based on the configuration, the model checkpoints are being written into the `--out_dir` directory `out-premchand-char`. So once the training finishes we can sample from the best model by pointing the sampling script at this directory:

```sh
python sample.py --out_dir=out-premchand-char
```

This generates a few samples, for example:
पुकारे--पास जढ़ी, सोदों में की हाथ कक यह सकतली में सारे को उनके पहच ड़ाला जाना ने से की होगा, बडबूर भी नहीं मसंत हुए थी हो?
पानात है मक कहीं इतनी कोई वह चार बोली का एक वह हम से ककर के दार से का बायद करने पास के िारा मेरेि तक ऊै।
उस।
जालपा सेिक बोले पासी को तुम उन्हें जी ब का िहती थी हुई िजर कर म कोई बूसर का िलखिा है है।
वह वह सकार माल कर मदया का खेश के आड़े आ था, किल से समए बात गहन-भी मु ंशीबर बड़ी है, जैस ठी-प़ि ग्या ने भी आपका पड़ा है िक पास नही और म्यान ही ह, ककुछ खड़े और मि तक िकहां जाता देग
---------------

बाहँ तक सलया खूँट ना पदु ं आता था ले िनकल बड़ गया।
एक सारा यह क दूसर को का रूपये साथो, तुमसी लेखसे के ले बाते हूंगा हूं, सुधी।
वहार पर चुवदि उसका उनका अमर्तु क्मलाम काल बाद धि नहीं लगती है।
दगधीन कु ही मकिर का िन कके सामनें उनके एक िपास में िनखल जाना कर मलया काम आदिी कर िहे क्यों की भर वह हो उससके उनकी मन सबसा में परका और भी जाते है , लोड जाते, दान हुए से बश को सुनन के क्या, मुझ नहीं त अलान शूर थी, उसका एक िलनट के सीवत्य थी।
उनके बोदां में से भी काल ककतने समझी, इस पहन र में कोछबोल से हो शास हुए क
---------------

बैठ उनका स्व खाश्जन के प्रसदय को मनो।
पाेपुी समझे प्रकु:कन पाप नार भी थी, रात हो ककहा, मैं इस उसके दसमुच्छ तो क्यों हो हो में आप से चली जानी, िह िजयाइतन क्मस करती थू बढ़े ?
मांग है, जभी िनाद है, तो यह िब-जी सुख से साहब बाू तो मारा कर सुनी बुलार-प्रे का बहुत मकया था।
म एक आई गए सुन के मल में जाते हैं बेच नकी के कही समता हँ?
मिरा ग पार्चि हुए क िगर बचार को िनमरकार कर णामस्त नहीं, कक्छा घर में न एक वह कर उसका सूर-दन िारा कका पहला को बात है हमारा की िचन नार न मकर त्र उन्हें ने देवतयस कर बारे पापर री
---------------

िलिमलान ज़ींग देख से और पह सब सारत बेचर्च ल है, रतनों उसकी ब तक मल से कहा, िससर समझता हूं के सपर इतना ले जाएगली ककी चाम समय कहीं मैं ककोई ही हुए दिखि हुआ िमक बने कोई बत बाड़ा था था।
तुम तुम थी, रमा सार चोरों ने कोई आंधत नही हो के न रहीं ह कके तो दो बाडी आए, तो मारी साथ कक को कै मल में क्षणा में अपन पर का बोला लग जाए ?
जालप हले पास भी तार मनोट का झोकर को अव व्यों से भी हैं, तो दू न कोई जाती हू पर िाश्वार मान पर चुकास तक कर हम िसर होगा।
यह ककही है, मैं कक हब क्या तो मकल सहिाया के पर ने उसकाल माल उ
---------------

रजात है पड़ा ह गया।
सार कद्वार कर माली ती है, 
चुभी कर ववसमारु:ख को के रदह मारे बह दूसर लाया जाने की समला पाप कर अपनी के नराश के के समिला िक तुवादि कर भी थी।
िजसका पहलाम करक्यार का क-हाँ न रखि रहा गईं, प्रतुब आ हिें रही है।
बेटा से दी थी, न एक  ऊूँ, बोली बैठने ककदर मानोटी थी, मानो का शशार के ‘मुि शान िकक्षा के खू था, तवह सोच था।
यह तुम्हारे में नहीं हला करना कर उसकी मुझे द गश्जा हृदयार सढ़ ने लगी थी।
इस क्यास-अभ्य उस िोता।
रमा का और सब लल कर ने का पर के खोल से सप्रया कर काए कक पु सोधी से के कक अ
---------------

दी जाकर हुई दूसमय हो गा।
अब क्य बुलाये थ, ककिर कहती है।
मैं उसने हम उसना उसके पाप है जान-म नही िक तरता है डरहे हाथ कक साहब तक जाया।
खिनों िलखना के ककाल िबन के आने तुम कार उन्हें ने का िदिस्वास मदको सब हुआ होता है।
तोई ने जाना परड़े कुल साहब ले िकर न कर साथ- आपका मचार आनने जो और सब लाया, अरू
इतन में की अपने गू गाया पहली हो गयीं कर जात था, तुम्हें हैं, पास, प्रसिी ख में सौ न भर वह हुए जैसिय नीचाब नहीं िैं दे।
मुझिोरी के पाते हो कहा हुए सालन चल जाते है, ह भैत बजाऊंगा, जी को ककुछ मुद्रत और का नहीं स
---------------

पर का तुम् सूक रतो उपसे प्रत यह रहुए मेरे असमान की जो जाएेगे?
अब ल जाते हुए ही लाया है।
ज्ञादारई कर तो िमलाजी के ही का सके की सामद्धिन से हो, 'सबस कैचा से कमलेने ही गई, हमसेराई है।
माला--क्या सुनध पर ने लग ने उसनके उनका जलू से के वल जा कर कक्या बार देते ही चल हो था।
पास रम सभी अपनी संसला में से धध न के बडये होते थे।
जब राफी नहीं से दो-िवपकार माना के आव मद जानों कर सूरक-जाकरने समकमला से जो से हुए भी नहीं हो सकदिर नहीं है।’ िकतश्मत नहीं भी तो हँ पहले, भाग हो, ‘जो कही आका कहीं है।
वह सककतवा रहा--ककएक <UNK> <UNK> । ’ रमानाथ — <UNK> एक <UNK> , यह <UNK> । ’ रमानाथ — <UNK> ? ’ रमानाथ — <UNK> ! ’ रमेश - - - - - - - - तुमने कल से इतना जल्दि पीने जाती हूं । ‘ रमा ने मुस्कराकर कहा , तुम्हें चकमा देते , तो क्या क्या जवाब न लेने से कोई उपाय है , जो रूपये होते हैं । अभी क्या है ? ’ दियानाथ — <UNK> , यह कहते हो , मेरे िदिल के ही न कर सकता है , तो <UNK> । ’ रमानाथ — <UNK> नहीं , तो तुम दिाम ? ’ दियानाथ — <UNK> को लेकर क्या चीजें आया , तो <UNK> । ’ रमानाथ — <UNK> के िलए दिो रमेश - - - - - - - - - - - <UNK> ? ’ जालपा — <UNK> है , तुम तो क्या मतलब ! ’ रमा ने <UNK> , मगर दियानाथ — <UNK> । मेरे पास न लू रमेश - - - - - - - - - — यह हाल मालूम होता , तो मुझिे <UNK> के पीछे लौटा देना तो आप खुशी से वादे पर िरश्वत के िलए कोई न था तुम्हें कुछ लौटा दे नहीं आता । अभी तक तुम्हारा िचता न देते । जैसे दिो - हां , तो मैं अब औरतें िमलने लग जाय , वह <UNK> की क्या ? ’ रमानाथ — <UNK> तुम्हें व्यथर्च है , जब से कहा , तो , िफर तो और जो उपाय नहीं तो शायदि इसी तरह महीने में आप एक चीजें तो व्यथर्च में एक चीज़ ले लीिजएगा , तो दिो - सात सौ रूपये ही में इस <UNK> । ‘ जालपा — <UNK> <UNK> का सबसे अिधक <UNK> - <UNK> , उसी िदिन तो कोई िकसी से पांच रूपयों में <UNK> । यह तो मुझिे मुझिे क्या एक िदिन नहीं होता । िफर िकसी <UNK> नहीं से खुदि बडा होगा ? ’ ंह उधरकर पोंछते हुए ? ’ दियानाथ - - ओढ़ने के िलए इसमें तो शायदि तीन हज़ार का नाम तो नहीं , तो सवेरा हो , तो नहीं । हां , तो मैंने कुछ नहीं , तो नहीं , जो तुम ज़रा सब तमाशा नहीं दे दिो । इससे कोई गहनों के ओढ़ने के <UNK> के कंगन नहीं । ’ रमा ने पूरा होता । ‘ जालपा — <UNK> , तो तुम अपने दिाम से कह देना । ’ रमा ने <UNK> जाते , हमें <UNK> , तो गुिडयों के िक वह तो शायदि यहां के िलए यहां नहीं । लेिकन दिो - तो बताइए , लेिकन शायदि मुझिे अज़ी नहीं हूं ? ’ जालपा — पांच रूपये तो नगदि क्यों एक चीज़ तो <UNK> , तो <UNK> । लूटने से कुछ <UNK> में <UNK> की <UNK> , मगर दिो - <UNK> - चार गहने बंदि कर सकता हूं , कभी <UNK> , बूढ़े - चार िदिन से वह तो चार िदिन हर दिो - तीन - एक चीज़ से यह जडाऊ िदिन <UNK> <UNK> । कहां से कुछ रूपये िमल
---------------
एक <UNK> में सभी उसके संप णद जीवन - चार <UNK> बन जाती है , यह एक चुटकी भर कर <UNK> करते <UNK> - सब <UNK> थीं । रायसाहब का कोई <UNK> - सा रस - भर - कभी <UNK> से भी दस - तले अपने भजन ह चुका था । न करिे हैं , इसनलए नहीं । रायसाहब ने कु छ न था । गोबर चमार । जब अपना <UNK> में कहा - एक गाय है । रुपए तो ककसी रुपए <UNK> में हम नसर पर जा कर द ध कोई <UNK> नहीं , तो किर , वह गऊ से बड़ी मुनककल से भेंट करना , तो यही वह छोटी - <UNK> से - कभी न करें , तो मैंने भी नहीं , वह चाहे <UNK> । धननया ने दो । होरी पर <UNK> के नलए द ध ध नमल गई । गोबर के नलए <UNK> के नलए रुपए होगी । ककसी तरह - न जाने से लौटे । किर द सरा नाूँद भी न <UNK> देख कर बोले - पीछे तो देखा , मैं तो उसके हाथ - अगर वह पछाईं गाय नमल जाय । अच्छा , तो क्या ? ककसी के - राम - अब तक न कर उठा कर कहा - भर की कमी है ? राम के के नलए तरस कर दो , तो रोज - - - वस का काम के पास <UNK> । नाटे खी जाते हैं । गाय तो बछवे ही । यही क्या कमी है , तो <UNK> - कै से <UNK> कर कोई द सरा आदमी , तो वह भ सा <UNK> - <UNK> पर भी तो मारे मज री है । ' मुझसे क्यों ? रुपए माूँगता ? रुपए तो रुपए में कु छ मदद तो इतनी िु ू ' नहीं तो रुपए खी हैं , ऐसा <UNK> । ' प णाफ ने नेवता देने लगी । ' नहीं है , तो मैंने मेरे घर के घर में कु छ न मानोगी तो वहाूँ का काम करता है । ' मुझसे ले कदया , तो अपने मन - पाूँच सेर द ध पड़िी थी । ' यह तो रुपए क्या ' ' <UNK> । ' ' बहुि देख कर कहा - <UNK> में ककसी की गई ? ' नहीं , वह भी कोई कु छ स झी , लेककन क्यों हो , मेरा मन - ' क्या यही <UNK> आज कु छ नहीं होता , <UNK> है ? ' कमलाप्रसाद - ' ' इस <UNK> जाना । ' न कोई <UNK> में कोई बात है । ' बदरीप्रसाद ने इस काम ही िो शायद उसके मु ाँह पानी की उसे अप्रसन्न करके बोली - ' बस , मेरी क्या िुमसे कोई बाि कही - ' क्या समझ लो , मैं िुम्हें जािी है । ' क्या मैं <UNK> । ' क्या कभी बड़ी <UNK> हो । ' बदरीप्रसाद ने गदगद कं ठ नहीं समझा । ' बदरीप्रसाद - ' नहीं आिा है , और दो - ' िुम क्यों रखिी हो प णाफ , मैं मेरे घर में कहा - '

---------------

इससे कहता है।
कभू खबुर के देश को इतना पर में ओर रूपुष्ट  र को वहां से उत ठहो हो लगा।
अगर का समानो से होगा, मिश्ट की की दिम कक्यों ह ।
इस िगर उत्त मसय आज भी काने पुट कर को िमर्च हो चली- िबड़ा।
बोल करते हुए देख लगा मु ंशीजी मकयल हों ककी जल मूिान के और यह साड़ िमजए ब तक्या अनिक बात है, सो पुरान कर अने खिला होगी।

```

lol  `¯\_(ツ)_/¯`. Not bad for a character-level model after 3 minutes of training on a GPU. Better results are quite likely obtainable by instead finetuning a pretrained GPT-2 model on this dataset (see finetuning section later).

**I only have a macbook** (or other cheap computer). No worries, we can still train a GPT but we want to dial things down a notch. I recommend getting the bleeding edge PyTorch nightly ([select it here](https://pytorch.org/get-started/locally/) when installing) as it is currently quite likely to make your code more efficient. But even without it, a simple train run could look as follows:

```sh
python train.py config/train_premchand_char.py --device=cpu --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0
```

Here, since we are running on CPU instead of GPU we must set both `--device=cpu` and also turn off PyTorch 2.0 compile with `--compile=False`. Then when we evaluate we get a bit more noisy but faster estimate (`--eval_iters=20`, down from 200), our context size is only 64 characters instead of 256, and the batch size only 12 examples per iteration, not 64. We'll also use a much smaller Transformer (4 layers, 4 heads, 128 embedding size), and decrease the number of iterations to 2000 (and correspondingly usually decay the learning rate to around max_iters with `--lr_decay_iters`). Because our network is so small we also ease down on regularization (`--dropout=0.0`). This still runs in about ~3 minutes, but gets us a loss of only 1.88 and therefore also worse samples, but it's still good fun:

```sh
python sample.py --out_dir=out-premchand-char --device=cpu
```
Generates samples like this:

```
GLEORKEN VINGHARD III:
Whell's the couse, the came light gacks,
And the for mought you in Aut fries the not high shee
bot thou the sought bechive in that to doth groan you,
No relving thee post mose the wear
```

Not bad for ~3 minutes on a CPU, for a hint of the right character gestalt. If you're willing to wait longer, feel free to tune the hyperparameters, increase the size of the network, the context length (`--block_size`), the length of training, etc.

Finally, on Apple Silicon Macbooks and with a recent PyTorch version make sure to add `--device=mps` (short for "Metal Performance Shaders"); PyTorch then uses the on-chip GPU that can *significantly* accelerate training (2-3X) and allow you to use larger networks. See [Issue 28](https://github.com/karpathy/nanoGPT/issues/28) for more.

## reproducing GPT-2

A more serious deep learning professional may be more interested in reproducing GPT-2 results. So here we go - we first tokenize the dataset, in this case the [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/), an open reproduction of OpenAI's (private) WebText:

```sh
python data/openwebtext/prepare.py
```

This downloads and tokenizes the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. It will create a `train.bin` and `val.bin` which holds the GPT2 BPE token ids in one sequence, stored as raw uint16 bytes. Then we're ready to kick off training. To reproduce GPT-2 (124M) you'll want at least an 8X A100 40GB node and run:

```sh
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

This will run for about 4 days using PyTorch Distributed Data Parallel (DDP) and go down to loss of ~2.85. Now, a GPT-2 model just evaluated on OWT gets a val loss of about 3.11, but if you finetune it it will come down to ~2.85 territory (due to an apparent domain gap), making the two models ~match.

If you're in a cluster environment and you are blessed with multiple GPU nodes you can make GPU go brrrr e.g. across 2 nodes like:

```sh
# Run on the first (master) node with example IP 123.456.123.456:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
# Run on the worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

It is a good idea to benchmark your interconnect (e.g. iperf3). In particular, if you don't have Infiniband then also prepend `NCCL_IB_DISABLE=1` to the above launches. Your multinode training will work, but most likely _crawl_. By default checkpoints are periodically written to the `--out_dir`. We can sample from the model by simply `python sample.py`.

Finally, to train on a single GPU simply run the `python train.py` script. Have a look at all of its args, the script tries to be very readable, hackable and transparent. You'll most likely want to tune a number of those variables depending on your needs.

## baselines

OpenAI GPT-2 checkpoints allow us to get some baselines in place for openwebtext. We can get the numbers as follows:

```sh
$ python train.py config/eval_gpt2.py
$ python train.py config/eval_gpt2_medium.py
$ python train.py config/eval_gpt2_large.py
$ python train.py config/eval_gpt2_xl.py
```

and observe the following losses on train and val:

| model | params | train loss | val loss |
| ------| ------ | ---------- | -------- |
| gpt2 | 124M         | 3.11  | 3.12     |
| gpt2-medium | 350M  | 2.85  | 2.84     |
| gpt2-large | 774M   | 2.66  | 2.67     |
| gpt2-xl | 1558M     | 2.56  | 2.54     |

However, we have to note that GPT-2 was trained on (closed, never released) WebText, while OpenWebText is just a best-effort open reproduction of this dataset. This means there is a dataset domain gap. Indeed, taking the GPT-2 (124M) checkpoint and finetuning on OWT directly for a while reaches loss down to ~2.85. This then becomes the more appropriate baseline w.r.t. reproduction.

## finetuning

Finetuning is no different than training, we just make sure to initialize from a pretrained model and train with a smaller learning rate. For an example of how to finetune a GPT on new text go to `data/premchand` and run `prepare.py` to download the tiny premchand dataset and render it into a `train.bin` and `val.bin`, using the OpenAI BPE tokenizer from GPT-2. Unlike OpenWebText this will run in seconds. Finetuning can take very little time, e.g. on a single GPU just a few minutes. Run an example finetuning like:

```sh
python train.py config/finetune_premchand.py
```

This will load the config parameter overrides in `config/finetune_premchand.py` (I didn't tune them much though). Basically, we initialize from a GPT2 checkpoint with `init_from` and train as normal, except shorter and with a small learning rate. If you're running out of memory try decreasing the model size (they are `{'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}`) or possibly decreasing the `block_size` (context length). The best checkpoint (lowest validation loss) will be in the `out_dir` directory, e.g. in `out-premchand` by default, per the config file. You can then run the code in `sample.py --out_dir=out-premchand`:

```
THEODORE:
Thou shalt sell me to the highest bidder: if I die,
I sell thee to the first; if I go mad,
I sell thee to the second; if I
lie, I sell thee to the third; if I slay,
I sell thee to the fourth: so buy or sell,
I tell thee again, thou shalt not sell my
possession.

JULIET:
And if thou steal, thou shalt not sell thyself.

THEODORE:
I do not steal; I sell the stolen goods.

THEODORE:
Thou know'st not what thou sell'st; thou, a woman,
Thou art ever a victim, a thing of no worth:
Thou hast no right, no right, but to be sold.
```

Whoa there, GPT, entering some dark place over there. I didn't really tune the hyperparameters in the config too much, feel free to try!

## sampling / inference

Use the script `sample.py` to sample either from pre-trained GPT-2 models released by OpenAI, or from a model you trained yourself. For example, here is a way to sample from the largest available `gpt2-xl` model:

```sh
python sample.py \
    --init_from=gpt2-xl \
    --start="What is the answer to life, the universe, and everything?" \
    --num_samples=5 --max_new_tokens=100
```

If you'd like to sample from a model you trained, use the `--out_dir` to point the code appropriately. You can also prompt the model with some text from a file, e.g. ```python sample.py --start=FILE:prompt.txt```.

## efficiency notes

For simple model benchmarking and profiling, `bench.py` might be useful. It's identical to what happens in the meat of the training loop of `train.py`, but omits much of the other complexities.

Note that the code by default uses [PyTorch 2.0](https://pytorch.org/get-started/pytorch-2.0/). At the time of writing (Dec 29, 2022) this makes `torch.compile()` available in the nightly release. The improvement from the one line of code is noticeable, e.g. cutting down iteration time from ~250ms / iter to 135ms / iter. Nice work PyTorch team!

## todos

- Investigate and add FSDP instead of DDP
- Eval zero-shot perplexities on standard evals (e.g. LAMBADA? HELM? etc.)
- Finetune the finetuning script, I think the hyperparams are not great
- Schedule for linear batch size increase during training
- Incorporate other embeddings (rotary, alibi)
- Separate out the optim buffers from model params in checkpoints I think
- Additional logging around network health (e.g. gradient clip events, magnitudes)
- Few more investigations around better init etc.

## troubleshooting

Note that by default this repo uses PyTorch 2.0 (i.e. `torch.compile`). This is fairly new and experimental, and not yet available on all platforms (e.g. Windows). If you're running into related error messages try to disable this by adding `--compile=False` flag. This will slow down the code but at least it will run.

For some context on this repository, GPT, and language modeling it might be helpful to watch my [Zero To Hero series](https://karpathy.ai/zero-to-hero.html). Specifically, the [GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY) is popular if you have some prior language modeling context.

For more questions/discussions feel free to stop by **#nanoGPT** on Discord:

[![](https://dcbadge.vercel.app/api/server/3zy8kqD9Cp?compact=true&style=flat)](https://discord.gg/3zy8kqD9Cp)

## acknowledgements

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), my favorite Cloud GPU provider. Thank you Lambda labs for sponsoring nanoGPT!
