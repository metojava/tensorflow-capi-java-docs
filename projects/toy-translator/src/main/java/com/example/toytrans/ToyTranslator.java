package com.example.toytrans;

import com.example.tfcapi.*;

import java.nio.file.Path;
import java.util.Random;

/**
 * Project 16 - Toy machine translation between two invented languages.
 *
 * "English-like": SUBJECT ADVERB VERB OBJECT   (e.g. cat quietly sees dog)
 * "Kavish":       mirror-image word order and a different lexicon - every
 *                 source word has exactly one Kavish counterpart, and
 *                 Kavish reads the sentence back-to-front.
 *
 * Translation therefore = lexical substitution + full reordering, which a
 * single LSTM encoder with a dense decoder head learns end-to-end:
 * Input3D(4, 20) one-hot source planes -> LSTM(64) -> Dense(4*20).
 * Trained with Regressor (targets are flattened float one-hot planes),
 * decoded by per-position argmax, evaluated by exact-sentence accuracy.
 *
 * Symbol ids (both languages share the same 20-symbol id space):
 *   subjects/objects 0..7, adverbs 8..13, verbs 14..19.
 */
public final class ToyTranslator {

    static final int T = 4;
    static final int V = 20;

    // slot pools: subjects / objects share one noun list
    static final String[] EN_NOUN = {"cat", "dog", "bird", "mouse",
                                     "bear", "wolf", "fox", "owl"};
    static final String[] EN_ADV  = {"quickly", "slowly", "quietly",
                                     "loudly", "today", "again"};
    static final String[] EN_VERB = {"chases", "eats", "sees",
                                     "follows", "fears", "ignores"};
    static final String[] KV_NOUN = {"miao", "kova", "tiri", "nusu",
                                     "brak", "velu", "sinta", "oru"};
    static final String[] KV_ADV  = {"pika", "loma", "susa",
                                     "drev", "anta", "keli"};
    static final String[] KV_VERB = {"zumo", "hata", "veni",
                                     "roka", "fala", "juta"};

    public static void main(String[] args) throws Exception {
        int epochs = args.length > 0 ? Integer.parseInt(args[0]) : 150;

        // ---------- parallel corpus ----------------------------------------
        Random gen = new Random(42L);
        int n = 1600;                        // 80% train = 1280 = 40 x batch 32
        int[][] srcIds = new int[n][T];
        int[][] tgtIds = new int[n][T];
        String[][] enSent = new String[n][T];
        for (int i = 0; i < n; i++) {
            int s = gen.nextInt(EN_NOUN.length);            // 0..7
            int adv = gen.nextInt(EN_ADV.length);           // 0..5 -> id 8+
            int v = gen.nextInt(EN_VERB.length);            // 0..5 -> id 14+
            int o = gen.nextInt(EN_NOUN.length);            // 0..7
            srcIds[i] = new int[]{ s,
                    8 + adv,
                    14 + v,
                    o };
            tgtIds[i] = new int[]{ o,                       // object first
                    v,                                      // then verb
                    adv,                                    // then adverb
                    s };                                    // subject last
            enSent[i][0] = EN_NOUN[s];
            enSent[i][1] = EN_ADV[adv];
            enSent[i][2] = EN_VERB[v];
            enSent[i][3] = EN_NOUN[o];
        }

        int[][] parts = Datasets.trainTestSplit(n, 0.2, 7L);
        float[] trainX = toPlanes(srcIds, parts[0]);
        float[] testX = toPlanes(srcIds, parts[1]);
        float[] trainY = toPlanes(tgtIds, parts[0]);
        float[] testY = toPlanes(tgtIds, parts[1]);
        System.out.printf("%d sentence pairs; split %d/%d%n",
                n, parts[0].length, parts[1].length);

        // ---------- encoder-decoder translator ------------------------------
        try (Regressor net = new Regressor(Optimizer.adam(0.01f), 42L,
                new Input3D(T, V),
                new LSTM(64),
                new Dense(T * V))) {

            net.setShuffleSeed(42L);
            EarlyStopping es = EarlyStopping.valLoss(15, true);
            Regressor.RegressionHistory h = net.fit(trainX, trainY, epochs,
                    32, testX, testY, true, es);
            System.out.printf("ran %d/%d epochs - best val MSE %.6f at epoch %d%n",
                    h.epochsRun(), epochs, es.bestScore, es.bestEpoch + 1);

            // ---------- evaluate -------------------------------------------
            float[] out = net.predict(testX);
            long exact = 0, posHits = 0;
            for (int k = 0; k < parts[1].length; k++) {
                boolean allOk = true;
                for (int t = 0; t < T; t++) {
                    int pred = Metrics.argmax(out, k * T * V + t * V, V);
                    if (pred == tgtIds[parts[1][k]][t]) posHits++;
                    else allOk = false;
                }
                if (allOk) exact++;
            }
            System.out.printf("word accuracy      : %.4f%n",
                    (double) posHits / (parts[1].length * (long) T));
            System.out.printf("exact translations : %.4f (%d/%d)%n",
                    (double) exact / parts[1].length, exact, parts[1].length);

            // ---------- demo: translate unseen sentences --------------------
            System.out.println("\nDemo translations (unseen sentences):");
            for (int k = 0; k < 4 && k < parts[1].length; k++) {
                int idx = parts[1][k];
                StringBuilder e = new StringBuilder("EN : ");
                StringBuilder kv = new StringBuilder("gold: ");
                StringBuilder pr = new StringBuilder("pred: ");
                for (int t = 0; t < T; t++) {
                    e.append(enSent[idx][t]).append(' ');
                    kv.append(kvWord(srcIds[idx], t)).append(' ');
                    pr.append(decodeWord(Metrics.argmax(out,
                            k * T * V + t * V, V), t)).append(' ');
                }
                System.out.println(e);
                System.out.println(kv);
                System.out.println(pr);
                System.out.println();
            }

            net.saveTo(Path.of("target", "toy-translator.bin"));
            System.out.println("Saved checkpoint target/toy-translator.bin");
        }
    }

    /* ------------------------- helper methods --------------------------- */

    /** Kavish word shown at target position t for a source sentence. */
    static String kvWord(int[] src, int t) {
        return switch (t) {
            case 0 -> KV_NOUN[src[3]];          // object first
            case 1 -> KV_VERB[src[2] - 14];     // then verb
            case 2 -> KV_ADV[src[1] - 8];       // then adverb
            default -> KV_NOUN[src[0]];         // subject last
        };
    }

    /** Maps a predicted target symbol id back to a Kavish word.
     *  Target ids are raw pool indices: nouns 0..7, adverbs/verbs 0..5. */
    static String decodeWord(int id, int t) {
        return switch (T - 1 - t) {
            case 0 -> KV_NOUN[id];          // t=3 subject
            case 1 -> KV_ADV[id];           // t=2 adverb
            case 2 -> KV_VERB[id];          // t=1 verb
            default -> KV_NOUN[id];         // t=0 object
        };
    }

    static float[] toPlanes(int[][] ids, int[] rows) {
        float[] out = new float[rows.length * T * V];
        for (int i = 0; i < rows.length; i++)
            for (int t = 0; t < T; t++)
                out[i * T * V + t * V + ids[rows[i]][t]] = 1f;
        return out;
    }

    private ToyTranslator() {}
}
