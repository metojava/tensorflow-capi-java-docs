package com.example.transducer;

import com.example.tfcapi.*;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;

/**
 * Project 15 - Sequence-to-sequence transduction with a single
 * encoder-decoder model that performs TWO tasks, selected by a control
 * token in position 0 of the input:
 *
 *   [REV]  d1..d6 -> d6..d1        (reverse)
 *   [SORT] d1..d6 -> ascending    (sort)
 *
 * Architecture = Pattern A from the NLP docs: Input3D(9,12) one-hot planes
 * -> LSTM(64) encoder -> Dense(9*12) dense decoder head. Trained by
 * Regressor because the targets are flattened float one-hot planes; decoded
 * by per-position argmax. Demonstrates task conditioning, exact-sequence
 * metrics and EarlyStopping.
 */
public final class SequenceTransducer {

    static final int P = 6;                 // payload length
    static final int T = P + 1;             // + control token position
    static final int V = 12;                // digits 0..9, REV=10, SORT=11
    static final int REV = 10, SORT = 11;

    public static void main(String[] args) throws Exception {
        int epochs = args.length > 0 ? Integer.parseInt(args[0]) : 250;

        // ---------- 1. Seeded parallel data --------------------------------
        Random gen = new Random(42L);
        int n = 1600;                        // 80% train = 1280 = 40 x batch 32
        float[] srcX = new float[n * T * V];
        float[] tgtY = new float[n * T * V];
        int[][] seq = new int[n][P];
        boolean[] isSort = new boolean[n];
        for (int i = 0; i < n; i++) {
            isSort[i] = gen.nextBoolean();
            for (int t = 0; t < P; t++) seq[i][t] = gen.nextInt(10);
            int[] out = isSort[i] ? sorted(seq[i]) : reversed(seq[i]);
            int ctrl = isSort[i] ? SORT : REV;
            put(srcX, i, 0, ctrl);
            for (int t = 0; t < P; t++) put(srcX, i, t + 1, seq[i][t]);
            put(tgtY, i, 0, ctrl);
            for (int t = 0; t < P; t++) put(tgtY, i, t + 1, out[t]);
        }

        int[][] parts = Datasets.trainTestSplit(n, 0.2, 7L);
        float[] trainX = Datasets.gatherRows(srcX, parts[0], T * V);
        float[] testX = Datasets.gatherRows(srcX, parts[1], T * V);
        float[] trainY = Datasets.gatherRows(tgtY, parts[0], T * V);
        float[] testY = Datasets.gatherRows(tgtY, parts[1], T * V);
        System.out.printf("%d pairs (%d reverse / %d sort); split %d/%d%n",
                n, count(isSort, false), count(isSort, true),
                parts[0].length, parts[1].length);

        // ---------- 2. Encoder-decoder model -------------------------------
        try (Regressor net = new Regressor(Optimizer.adam(0.01f), 42L,
                new Input3D(T, V),
                new LSTM(96),
                new Dense(T * V))) {

            net.setShuffleSeed(42L);
            EarlyStopping es = EarlyStopping.valLoss(25, true);
            Regressor.RegressionHistory h = net.fit(trainX, trainY, epochs,
                    32, testX, testY, true, es);
            System.out.printf("ran %d/%d epochs - best val MSE %.6f at epoch %d%n",
                    h.epochsRun(), epochs, es.bestScore, es.bestEpoch + 1);

            // ---------- 3. Decode + score per task --------------------------
            float[] out = net.predict(testX);
            long revExact = 0, srtExact = 0, revN = 0, srtN = 0;
            long digitHits = 0, digitTotal = 0;
            int demoRev = -1, demoSrt = -1;
            for (int k = 0; k < parts[1].length; k++) {
                boolean sortTask = isSort[parts[1][k]];
                boolean allOk = true;
                for (int t = 0; t < T; t++) {
                    int pred = Metrics.argmax(out, k * T * V + t * V, V);
                    int truth = Metrics.argmax(testY, k * T * V + t * V, V);
                    digitTotal++;
                    if (pred == truth) digitHits++; else allOk = false;
                }
                if (sortTask) { srtN++; if (allOk) srtExact++; if (demoSrt < 0) demoSrt = k; }
                else          { revN++; if (allOk) revExact++; if (demoRev < 0) demoRev = k; }
            }
            System.out.printf("digit accuracy : %.4f%n",
                    (double) digitHits / digitTotal);
            System.out.printf("exact REVERSE  : %.4f (%d/%d)%n",
                    (double) revExact / revN, revExact, revN);
            System.out.printf("exact SORT     : %.4f (%d/%d)%n",
                    (double) srtExact / srtN, srtExact, srtN);

            showDemo("REVERSE", testX, testY, out, demoRev, T, V);
            showDemo("SORT", testX, testY, out, demoSrt, T, V);

            net.saveTo(Path.of("target", "transducer.bin"));
            System.out.println("Saved checkpoint target/transducer.bin");
        }
    }

    /* ------------------------- helper methods --------------------------- */

    static void put(float[] plane, int row, int pos, int id) {
        plane[row * T * V + pos * V + id] = 1f;
    }

    static int[] reversed(int[] a) {
        int[] r = a.clone();
        for (int i = 0, j = r.length - 1; i < j; i++, j--) {
            int t = r[i]; r[i] = r[j]; r[j] = t;
        }
        return r;
    }

    static int[] sorted(int[] a) {
        int[] r = a.clone();
        Arrays.sort(r);
        return r;
    }

    static long count(boolean[] flags, boolean v) {
        long c = 0;
        for (boolean b : flags) if (b == v) c++;
        return c;
    }

    static void showDemo(String label, float[] x, float[] y, float[] pred,
                         int k, int T, int V) {
        StringBuilder in = new StringBuilder("[");
        StringBuilder tg = new StringBuilder();
        StringBuilder pr = new StringBuilder();
        for (int t = 0; t < T; t++) {
            int pi = Metrics.argmax(x, k * T * V + t * V, V);
            in.append(pi == REV ? "REV" : pi == SORT ? "SRT" : pi)
              .append(t < T - 1 ? "," : "]");
            if (t > 0) {
                tg.append(Metrics.argmax(y, k * T * V + t * V, V));
                pr.append(Metrics.argmax(pred, k * T * V + t * V, V));
                if (t < T - 1) { tg.append(' '); pr.append(' '); }
            }
        }
        System.out.println(label + " input=" + in + " target=" + tg
                + " predicted=" + pr);
    }

    private SequenceTransducer() {}
}
