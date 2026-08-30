package com.example.newstopic;

import com.example.tfcapi.*;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Project 13 - News topic classification: multi-class NLP end-to-end.
 *
 * Synthetic headlines are generated from four topic vocabularies (sport,
 * tech, weather, finance) plus neutral filler words. The pipeline mirrors
 * real NLP: vocabulary from TRAIN tokens only, tokens to ids (UNK/PAD),
 * fixed-length encoding - and then TWO architectures are compared:
 *
 * 1. Embedding bag-of-words: Input3D(T,1) -> Embedding(V,16) -> Dense head.
 *    Fast, ignores word ORDER (the Embedding output is flattened).
 * 2. Order-aware sequence model: Input3D(T,V) one-hot token planes
 *    -> LSTM(24) -> Dense head. Slower, reads the headline word by word.
 *
 * Demonstrates: seeded data + Datasets.trainTestSplit/gatherRows/gatherValues,
 * leakage-free vocabulary, EarlyStopping, and Metrics.printReport per class.
 */
public final class NewsTopicClassifier {

    /** Fixed sequence length every headline is encoded to. */
    static final int T = 10;
    /** Special ids reserved before any corpus word. */
    static final int UNK = 0;
    static final int PAD = 1;
    private static final String[] TOPICS = {"sport", "tech", "weather", "finance"};

    private static final String[][] KEYWORDS = {
        { "goal", "striker", "match", "league", "coach", "cup", "penalty",
          "tournament", "defender", "kickoff", "stadium", "champions" },
        { "software", "chip", "startup", "app", "cloud", "robot", "battery",
          "quantum", "algorithm", "smartphone", "server", "launch" },
        { "storm", "rainfall", "heatwave", "forecast", "snowfall", "drought",
          "thunderstorms", "humidity", "blizzard", "sunshine", "floods", "winds" },
        { "stocks", "inflation", "market", "earnings", "bonds", "merger",
          "dividend", "recession", "currency", "traders", "shares", "bank" }
    };

    private static final String[] FILLER = {
        "the", "a", "new", "big", "local", "early", "late", "major", "small",
        "city", "team", "report", "officials", "week", "year", "night",
        "morning", "group", "panel", "plans", "warns", "says", "wins", "faces"
    };

    public static void main(String[] args) throws Exception {
        int epochs = args.length > 0 ? Integer.parseInt(args[0]) : 150;

        // ---------- 1. Generate synthetic headlines -----------------------
        Random gen = new Random(42L);
        int n = 1600;                                  // 80% train = 1280 = 40 x batch 32
        String[] text = new String[n];
        int[] label = new int[n];
        for (int i = 0; i < n; i++) {
            label[i] = gen.nextInt(TOPICS.length);
            text[i] = headline(gen, label[i]);
        }

        // ---------- 2. Seeded split via Datasets --------------------------
        int[][] parts = Datasets.trainTestSplit(n, 0.2, 7L);
        int[] trIdx = parts[0], teIdx = parts[1];
        System.out.printf("Generated %d headlines across %d topics; split %d/%d%n",
                n, TOPICS.length, trIdx.length, teIdx.length);

        // ---------- 3. Vocabulary from TRAIN tokens only -------------------
        Map<String, Integer> freq = new HashMap<>();
        for (int i : trIdx)
            for (String w : tokenize(text[i]))
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
        int v = vocab.size();
        System.out.println("distinct words=" + freq.size()
                + " vocabSize=" + v + " (built on TRAIN only)");

        // ---------- 4. Encode both splits ----------------------------------
        int nTrain = trIdx.length, nTest = teIdx.length;
        float[] trainX = new float[nTrain * T];
        int[] trainY = Datasets.gatherValues(label, trIdx);
        float[] testX = new float[nTest * T];
        int[] testY = Datasets.gatherValues(label, teIdx);
        fillEncoded(text, trIdx, vocab, trainX);
        fillEncoded(text, teIdx, vocab, testX);
        System.out.println("example: \"" + text[trIdx[0]] + "\" -> "
                + showIds(trainX, 0));

        double accEmbedding, accLstm;

        // ---------- 5. Model A: Embedding bag-of-words ----------------------
        System.out.println("\n=== Model A: Embedding(vocab," + 16 + ") + Dense ===");
        try (Sequential net = new Sequential(Optimizer.adam(0.01f), 42L,
                new Input3D(T, 1),
                new Embedding(v, 16),
                new Dense(32), new ReLU(),
                new Dropout(0.2f),
                new Dense(TOPICS.length))) {

            net.setShuffleSeed(42L);
            EarlyStopping es = EarlyStopping.valLoss(20, true);
            AbstractClassifier.History h = net.fit(trainX, trainY, epochs, 32,
                    testX, testY, true, es);
            System.out.printf("ran %d/%d epochs, best val loss %.4f%n",
                    h.epochsRun(), epochs, es.bestScore);
            accEmbedding = accuracy(net, testX, testY);
        }

        // ---------- 6. Model B: LSTM over one-hot token planes -------------
        // Each sample becomes T x V one-hot planes (T*V floats), letting the
        // LSTM read one token vector per time step.
        System.out.println("\n=== Model B: one-hot Input3D(T," + v + ") -> LSTM(24) ===");
        float[] trainOh = new float[nTrain * T * v];
        float[] testOh = new float[nTest * T * v];
        fillOneHot(text, trIdx, vocab, v, trainOh);
        fillOneHot(text, teIdx, vocab, v, testOh);
        try (Sequential net = new Sequential(Optimizer.adam(0.01f), 43L,
                new Input3D(T, v),
                new LSTM(24),
                new Dense(32), new ReLU(),
                new Dense(TOPICS.length))) {

            net.setShuffleSeed(43L);
            EarlyStopping es = EarlyStopping.valLoss(20, true);
            AbstractClassifier.History h = net.fit(trainOh, trainY, epochs, 32,
                    testOh, testY, true, es);
            System.out.printf("ran %d/%d epochs, best val loss %.4f%n",
                    h.epochsRun(), epochs, es.bestScore);
            accLstm = accuracy(net, testOh, testY);

            // ---------- 7. Demo on NEW unseen headlines --------------------
            System.out.println("\nDemo predictions on NEW headlines (order-aware model):");
            String[] demo = {
                "the striker scored a goal after the storm delay",
                "a startup says its quantum algorithm beats big cloud servers",
                "heatwave warnings as humidity and winds rise all week",
                "traders dump shares as inflation hits the bank earnings"
            };
            for (String sent : demo) {
                float[] one = new float[T * v];
                int[] ids = encode(sent, vocab);
                for (int t = 0; t < T; t++)
                    one[t * v + ids[t]] = 1f;
                float[] pr = net.predict(one);
                int pred = Metrics.argmax(pr, 0, TOPICS.length);
                System.out.printf("\"%s\"%n  -> %s%n", sent, TOPICS[pred]);
            }

            net.saveTo(Path.of("target", "news-topic-model.bin"));
            System.out.println("Saved checkpoint target/news-topic-model.bin");
        }

        System.out.printf("%nTest accuracy: embedding-bag %.3f | lstm %.3f%n",
                accEmbedding, accLstm);
    }

    /* ------------------------- helper methods --------------------------- */

    /** Builds a headline: 2-4 topic keywords shuffled among 3-6 fillers. */
    static String headline(Random gen, int topic) {
        String[] kw = KEYWORDS[topic];
        List<String> w = new ArrayList<>();
        int nKw = 2 + gen.nextInt(3);
        for (int i = 0; i < nKw; i++) w.add(kw[gen.nextInt(kw.length)]);
        int fillers = 3 + gen.nextInt(4);
        for (int i = 0; i < fillers; i++)
            w.add(FILLER[gen.nextInt(FILLER.length)]);
        Collections.shuffle(w, gen);
        return String.join(" ", w);
    }

    static List<String> tokenize(String sentence) {
        String clean = sentence.toLowerCase().replaceAll("[^a-z]+", " ").trim();
        List<String> out = new ArrayList<>();
        for (String tok : clean.split("\\s+"))
            if (!tok.isEmpty()) out.add(tok);
        return out;
    }

    static int[] encode(String sentence, Map<String, Integer> vocab) {
        int[] ids = new int[T];
        for (int t = 0; t < T; t++) ids[t] = PAD;
        List<String> toks = tokenize(sentence);
        for (int t = 0; t < Math.min(toks.size(), T); t++)
            ids[t] = vocab.getOrDefault(toks.get(t), UNK);
        return ids;
    }

    static void fillEncoded(String[] text, int[] idx, Map<String, Integer> vocab,
                            float[] out) {
        for (int i = 0; i < idx.length; i++) {
            int[] ids = encode(text[idx[i]], vocab);
            for (int t = 0; t < T; t++) out[i * T + t] = ids[t];
        }
    }

    /** One-hot variant: every token becomes a length-V plane per time step. */
    static void fillOneHot(String[] text, int[] idx, Map<String, Integer> vocab,
                           int v, float[] out) {
        java.util.Arrays.fill(out, 0f);
        for (int i = 0; i < idx.length; i++) {
            int[] ids = encode(text[idx[i]], vocab);
            for (int t = 0; t < T; t++) out[i * T * v + t * v + ids[t]] = 1f;
        }
    }

    static String showIds(float[] x, int row) {
        StringBuilder sb = new StringBuilder("[");
        for (int t = 0; t < T; t++)
            sb.append((int) x[row * T + t]).append(t < T - 1 ? "," : "]");
        return sb.toString();
    }

    /** Cross-entropy loss, accuracy and confusion report for a classifier. */
    static double accuracy(Sequential net, float[] x, int[] y) {
        float[] probs = net.predict(x);
        int correct = 0;
        double loss = 0;
        int[][] conf = new int[TOPICS.length][TOPICS.length];
        for (int i = 0; i < y.length; i++) {
            int pred = Metrics.argmax(probs, i * TOPICS.length, TOPICS.length);
            conf[y[i]][pred]++;
            if (pred == y[i]) correct++;
            loss += -Math.log(Math.max(probs[i * TOPICS.length + y[i]], 1e-12));
        }
        Metrics.printReport(loss / y.length, (double) correct / y.length,
                conf, TOPICS);
        return (double) correct / y.length;
    }

    private NewsTopicClassifier() {}
}
