package com.example.charlm;

import com.example.tfcapi.*;

import java.util.Random;

/**
 * Project 14 - Character-level language model that GENERATES text.
 *
 * A sliding window of T=16 characters is fed as one-hot planes
 * (Input3D(16, V)) through an LSTM(48); a Dense(V) head classifies which
 * character comes NEXT - a plain classification task whose labels are the
 * next character ids. Once trained, the model becomes generative: feed any
 * 16-character seed, sample the predicted distribution, append, slide,
 * repeat - and it writes new headline-like text character by character.
 *
 * Demonstrates: sliding-window dataset construction, temperature sampling,
 * next-character evaluation, EarlyStopping.
 */
public final class CharLmTextGenerator {

    static final String CHARS = "abcdefghijklmnopqrstuvwxyz .\n";
    static final int T = 16;

    public static void main(String[] args) throws Exception {
        int epochs = args.length > 0 ? Integer.parseInt(args[0]) : 40;
        int v = CHARS.length();

        // ---------- 1. Seeded corpus of headline-style lines --------------
        Random gen = new Random(42L);
        StringBuilder corpus = new StringBuilder();
        for (int i = 0; i < 1200; i++) {
            corpus.append(headline(gen)).append(".\n");
        }
        String text = corpus.toString();
        System.out.printf("corpus: %d chars, alphabet=%d%n", text.length(), v);

        // ---------- 2. Sliding windows: T chars in -> next char out -------
        int stride = 4;
        int wMax = (text.length() - T - 1) / stride;
        float[] winX = new float[wMax * T * v];
        int[] winY = new int[wMax];
        int w = 0;
        for (int pos = 0; pos + T < text.length() && w < wMax; pos += stride) {
            for (int t = 0; t < T; t++)
                winX[w * T * v + t * v + id(text.charAt(pos + t))] = 1f;
            winY[w++] = id(text.charAt(pos + T));
        }
        int[][] parts = Datasets.trainTestSplit(w, 0.2, 7L);
        // trim the train split to an exact multiple of batch 128
        parts[0] = java.util.Arrays.copyOf(parts[0], parts[0].length / 128 * 128);
        System.out.println("training windows: " + parts[0].length);

        float[] trainX = Datasets.gatherRows(winX, parts[0], T * v);
        float[] testX = Datasets.gatherRows(winX, parts[1], T * v);
        int[] trainY = Datasets.gatherValues(winY, parts[0]);
        int[] testY = Datasets.gatherValues(winY, parts[1]);

        // ---------- 3. The tiny language model -----------------------------
        try (Sequential net = new Sequential(Optimizer.adam(0.01f), 42L,
                new Input3D(T, v),
                new LSTM(48),
                new Dense(v))) {

            net.setShuffleSeed(42L);
            EarlyStopping es = EarlyStopping.valLoss(8, true);
            AbstractClassifier.History h =
                    net.fit(trainX, trainY, epochs, 128, testX, testY, true, es);
            System.out.printf("ran %d/%d epochs - best val loss %.4f at epoch %d%n",
                    h.epochsRun(), epochs, es.bestScore, es.bestEpoch + 1);

            float[] probs = net.predict(testX);
            int hits = 0;
            for (int i = 0; i < testY.length; i++)
                if (Metrics.argmax(probs, i * v, v) == testY[i]) hits++;
            System.out.printf("next-char accuracy: %.4f%n",
                    (double) hits / testY.length);

            // ---------- 4. GENERATE text ---------------------------------
            net.setTraining(false);
            int seedPos = text.lastIndexOf("the ", text.length() / 2);
            char[] ctx = text.substring(seedPos, seedPos + T).toCharArray();
            System.out.println("\ngenerated (temp 0.8):");
            System.out.print("\"" + new String(ctx));
            Random sampler = new Random(7L);
            double temp = 0.8;
            for (int step = 0; step < 220; step++) {
                float[] one = new float[T * v];
                for (int t = 0; t < T; t++)
                    one[t * v + id(ctx[t])] = 1f;
                float[] p = net.predict(one);
                // temperature: sharpen (temp<1) or flatten (temp>1)
                double sum = 0;
                double[] wts = new double[v];
                for (int c = 0; c < v; c++) {
                    wts[c] = Math.pow(Math.max(p[c], 1e-9f), 1.0 / temp);
                    sum += wts[c];
                }
                double r = sampler.nextDouble() * sum;
                int next = v - 1;
                for (int c = 0; c < v; c++) {
                    r -= wts[c];
                    if (r <= 0) { next = c; break; }
                }
                System.out.print(CHARS.charAt(next));
                System.arraycopy(ctx, 1, ctx, 0, T - 1);
                ctx[T - 1] = CHARS.charAt(next);
                if (CHARS.charAt(next) == '\n' && step > 40) break;
            }
            System.out.println("\"");

            net.saveTo(java.nio.file.Path.of("target", "char-lm.bin"));
            System.out.println("Saved checkpoint target/char-lm.bin");
        }
    }

    static int id(char c) {
        int idx = CHARS.indexOf(c);
        return idx >= 0 ? idx : CHARS.indexOf(' ');
    }

    /** Headline-style line: topic-free keyword/filler mix with variety. */
    static String headline(Random g) {
        String[] words = {
            "goal", "striker", "storm", "market", "chip", "cloud", "coach",
            "inflation", "forecast", "startup", "league", "heatwave", "bank",
            "robot", "cup", "earnings", "blizzard", "server", "penalty",
            "the", "a", "new", "big", "local", "major", "small", "early",
            "wins", "warns", "says", "rises", "falls", "faces", "plans"
        };
        int len = 4 + g.nextInt(4);
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < len; i++) {
            sb.append(words[g.nextInt(words.length)]);
            sb.append(g.nextInt(100) < 70 ? ' ' : '.');
        }
        return sb.toString().trim();
    }

    private CharLmTextGenerator() {}
}
