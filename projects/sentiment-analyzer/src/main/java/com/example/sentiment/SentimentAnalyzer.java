package com.example.sentiment;

import com.example.tfcapi.*;

import java.nio.file.Path;
import java.util.Random;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Project 7 - Binary sentiment classification of short movie reviews.
 *
 * The corpus is hand-curated directly in code: 40 positive + 40 negative
 * one-line reviews. Words are mapped to integer ids (UNK=0, PAD=1, real words
 * from 2 up) and every sentence is encoded to a fixed length T=12 by
 * truncating or right-padding with PAD.
 *
 * Model: Input3D(12,1) -> Embedding(vocabSize,16) -> Dense(16) -> ReLU
 *        -> Dropout(0.2) -> Dense(2)
 *
 * The Embedding layer is a trainable lookup table; its output for a sentence
 * of T tokens with embDim=16 flattens to a [T*embDim] = 192-vector that the
 * Dense layers classify.
 */
public final class SentimentAnalyzer {

    /** Fixed sequence length every review is encoded to. */
    static final int T = 12;
    /** Special ids reserved before any corpus word. */
    static final int UNK = 0;
    static final int PAD = 1;
    private static final String[] NAMES = {"negative", "positive"};

    // -------------------- hand-curated corpus (authored for this project) ---
    // Sentiment-bearing words deliberately RECURRENCE across many sentences
    // so the classifier can learn them from only ~64 training reviews; the
    // remaining filler words supply vocabulary variety.
    static final String[] POSITIVE = {
        "this wonderful movie has brilliant acting and a great story",
        "a fantastic film with great acting and a wonderful story",
        "i loved this brilliant comedy great fun from beginning to end",
        "excellent acting a superb script and a wonderful elegant cast",
        "amazing visuals wonderful music and great performances",
        "one of the best films this year absolutely wonderful",
        "brilliant direction a great plot and perfect pacing",
        "a charming funny movie with great dialogue",
        "gorgeous cinematography great characters i loved it",
        "thrilling action a great cast and fantastic stunts",
        "a heartwarming story excellent acting and wonderful radiant music",
        "superb filmmaking an engaging drama simply perfect",
        "wonderful witty dialogue great fun throughout",
        "fantastic acting a brilliant script and stunning locations",
        "delightful charming and utterly engaging great cinema",
        "the best thriller this decade great and gripping",
        "great fun amazing stunts and a wonderful cast",
        "beautifully made excellent performances and wonderful uplifting music",
        "loved every minute of this brilliant funny film",
        "a perfect movie charming humor and a wonderful glorious heart",
        "stunning animation great songs and wonderful voice acting",
        "engaging characters brilliant writing and a perfect ending",
        "excellent storytelling superb tension and a great payoff",
        "an amazing experience wonderful from the opening scene",
        "wonderful chemistry great jokes and brilliant leads",
        "brilliant bold and fantastic a truly great film",
        "a gorgeous picture superb acting and great depth",
        "fun fast and fantastically entertaining i loved it",
        "a wonderful masterpiece amazing craft and great storytelling",
        "charming delightful full of heart simply wonderful",
        "great pacing stunning visuals and an excellent cast",
        "a brilliant comedy witty lines and perfect timing",
        "wonderful storytelling amazing music and great vivid messages",
        "superb thrilling and completely engaging great graceful cinema",
        "excellent twists great acting and a memorable finale",
        "a fun heartwarming adventure with a wonderful cast",
        "amazing costumes gorgeous sets and fantastic acting",
        "loved the brilliant story the great humor the charm",
        "wonderfully acted beautifully filmed simply great",
        "perfect entertainment a fantastic script and great acting"
    };

    static final String[] NEGATIVE = {
        "this terrible movie has awful acting and a boring story",
        "a dreadful film with awful acting and a boring grim script",
        "i hated every minute of this dull predictable mess",
        "terrible acting a weak script and an awful bleak cast",
        "awful visuals boring music and terrible performances",
        "one of the worst films this year absolutely awful",
        "clumsy direction a boring plot and terrible pacing",
        "a dull unfunny movie with bad dialogue",
        "ugly cinematography forgettable characters i hated it",
        "tedious scenes a weak cast and awful dreary action",
        "a boring story awful acting and annoying hollow music",
        "sloppy filmmaking a dull drama a complete waste",
        "terrible cringe dialogue flat jokes and awful pacing",
        "awful acting a terrible script and laughable effects",
        "dull slow and utterly boring a complete waste",
        "the worst thriller this decade dull and exhausting",
        "no fun at all clumsy stunts and an awful cast",
        "poorly made wooden performances and a weak insipid script",
        "hated this long boring film a total waste",
        "a painful movie forced humor zero charm",
        "cheap animation awful voice acting and bad songs",
        "flat characters lazy writing and a terrible ending",
        "terrible storytelling awful tension and a poor payoff",
        "an agonizing experience dull from the opening scene",
        "zero chemistry bad jokes and weak leads",
        "weak derivative and terrible a truly bad repulsive film",
        "an ugly picture bad acting and a boring plot",
        "slow stale and painfully boring i hated it",
        "a tedious disaster sloppy craft and awful pacing",
        "dull dreary cold a true disappointment",
        "bad pacing murky visuals and an awful cast",
        "a weak comedy flat lines and awful timing",
        "terrible storytelling awful music and confusing bleak messages",
        "sloppy dull and completely forgettable bad cinema",
        "predictable twists weak acting and a lame ending",
        "a tedious slog with unlikable characters and bad acting",
        "cheap costumes ugly sets and terrible acting",
        "hated the boring story the bad humor the mess",
        "badly acted poorly filmed completely unsatisfying",
        "simply awful entertainment a terrible script and bad acting"
    };

    public static void main(String[] args) throws Exception {
        // ---------- 1. Build the vocabulary from the corpus ---------------
        // id 0 = UNK (unknown word), id 1 = PAD (padding), then words sorted
        // by descending frequency (ties broken alphabetically for stability).
        List<String> sentences = new ArrayList<>();
        int[] labels = new int[POSITIVE.length + NEGATIVE.length];
        int s = 0;
        for (String p : POSITIVE) { sentences.add(p); labels[s++] = 1; }
        for (String n : NEGATIVE) { sentences.add(n); labels[s++] = 0; }

        Map<String, Integer> freq = new HashMap<>();
        for (String sent : sentences)
            for (String w : tokenize(sent))
                freq.merge(w, 1, Integer::sum);

        List<Map.Entry<String, Integer>> byFreq = new ArrayList<>(freq.entrySet());
        byFreq.sort((a, b) -> {
            int c = Integer.compare(b.getValue(), a.getValue());
            return c != 0 ? c : a.getKey().compareTo(b.getKey());
        });
        Map<String, Integer> vocab = new HashMap<>();
        vocab.put("<UNK>", UNK);
        vocab.put("<PAD>", PAD);
        for (Map.Entry<String, Integer> e : byFreq)
            vocab.put(e.getKey(), vocab.size());
        int vocabSize = vocab.size();
        System.out.println("corpus sentences=" + sentences.size()
                + " distinct words=" + freq.size() + " vocabSize=" + vocabSize);

        // ---------- 2. Encode every sentence to fixed length T ------------
        int n = sentences.size();
        float[] x = new float[n * T];
        for (int i = 0; i < n; i++) {
            int[] ids = encode(sentences.get(i), vocab);
            for (int t = 0; t < T; t++) x[i * T + t] = ids[t];
        }

        // ---------- 3. Seeded shuffle + 80/20 split ------------------------
        int[] order = seededOrder(n, 42);
        int nTrain = (int) Math.round(n * 0.8);
        float[] trainX = new float[nTrain * T];
        int[] trainY = new int[nTrain];
        float[] testX = new float[(n - nTrain) * T];
        int[] testY = new int[n - nTrain];
        for (int i = 0; i < n; i++) {
            if (i < nTrain) {
                System.arraycopy(x, order[i] * T, trainX, i * T, T);
                trainY[i] = labels[order[i]];
            } else {
                System.arraycopy(x, order[i] * T, testX, (i - nTrain) * T, T);
                testY[i - nTrain] = labels[order[i]];
            }
        }
        System.out.println("train=" + nTrain + " test=" + (n - nTrain));

        // ---------- 4. Build and train the model ---------------------------
        try (Sequential net = new Sequential(Optimizer.adam(0.02f),
                new Input3D(T, 1),
                new Embedding(vocabSize, 16),
                new Dense(16), new ReLU(),
                new Dropout(0.2f),
                new Dense(2))) {

            net.fit(trainX, trainY, 250, 16, testX, testY, true);

            // ---------- 5. Evaluate on the held-out 20% -------------------
            float[] probs = net.predict(testX);
            double loss = 0;
            int correct = 0;
            int[][] conf = new int[2][2];
            for (int i = 0; i < testY.length; i++) {
                int pred = Metrics.argmax(probs, i * 2, 2);
                conf[testY[i]][pred]++;
                if (pred == testY[i]) correct++;
                loss += -Math.log(Math.max(probs[i * 2 + testY[i]], 1e-12));
            }
            Metrics.printReport(loss / testY.length,
                    (double) correct / testY.length, conf, NAMES);

            net.saveTo(Path.of("target", "sentiment-model.bin"));
            System.out.println("Model saved to target/sentiment-model.bin");

            // ---------- 6. Demo on four NEW unseen sentences --------------
            String[] demo = {
                "a wonderful and brilliant film with great acting",
                "dull awful acting and a boring script what a waste",
                "gorgeous visuals charming characters i loved this movie",
                "the worst movie of the year terrible and painful"
            };
            System.out.println("\nDemo predictions on NEW sentences:");
            for (String sent : demo) {
                float[] one = new float[T];
                int[] ids = encode(sent, vocab);
                for (int t = 0; t < T; t++) one[t] = ids[t];
                float[] pr = net.predict(one);
                int pred = Metrics.argmax(pr, 0, 2);
                System.out.printf("\"%s\"%n  -> %s  P(negative)=%.3f P(positive)=%.3f%n",
                        sent, NAMES[pred], pr[0], pr[1]);
            }
        }
    }

    /* ------------------------- helper methods --------------------------- */

    /** Lowercase, strip punctuation, split on whitespace. */
    static List<String> tokenize(String sentence) {
        String clean = sentence.toLowerCase().replaceAll("[^a-z]+", " ").trim();
        List<String> out = new ArrayList<>();
        for (String w : clean.split("\\s+"))
            if (!w.isEmpty()) out.add(w);
        return out;
    }

    /** Map tokens to ids, truncate at T and pad the remainder with PAD. */
    static int[] encode(String sentence, Map<String, Integer> vocab) {
        int[] ids = new int[T];
        for (int t = 0; t < T; t++) ids[t] = PAD;
        List<String> toks = tokenize(sentence);
        for (int t = 0; t < Math.min(toks.size(), T); t++)
            ids[t] = vocab.getOrDefault(toks.get(t), UNK);
        return ids;
    }

    static int[] seededOrder(int n, long seed) {
        int[] o = new int[n];
        for (int i = 0; i < n; i++) o[i] = i;
        Random r = new Random(seed);
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1), t = o[i]; o[i] = o[j]; o[j] = t;
        }
        return o;
    }

    private SentimentAnalyzer() {}
}
