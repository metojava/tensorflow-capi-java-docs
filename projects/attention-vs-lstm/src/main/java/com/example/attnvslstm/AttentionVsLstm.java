package com.example.attnvslstm;

import com.example.tfcapi.*;

import java.nio.file.Path;
import java.util.Random;

/**
 * Project 17 - The task recurrent nets struggle with and attention eats for
 * breakfast: DUPLICATE DETECTION.
 *
 * Given a digit sequence, decide whether any symbol appears more than once.
 * The check is order-free and pairwise - every position must compare itself
 * against every other - which is exactly what self-attention computes in one
 * hop, while an LSTM has to carry comparisons through its hidden state.
 *
 * Two architectures, same data, same budget:
 *   A) mini-transformer: Input3D(T,V) -> SelfAttention x2 -> Flatten -> Dense
 *   B) recurrent:        Input3D(T,V) -> LSTM(32)     -> Dense
 *
 * NOTE: no positional encoding exists in the library yet, which is fine here
 * - duplicate detection is order-independent by definition.
 */
public final class AttentionVsLstm {

    static final int T = 8;
    static final int V = 10;

    public static void main(String[] args) throws Exception {
        int epochs = args.length > 0 ? Integer.parseInt(args[0]) : 300;

        // ---------- 1. Seeded balanced dataset ------------------------------
        // Needle-in-haystack retrieval: position 0 holds a PROBE digit; the
        // label asks whether that probe appears among the T-1 payload digits.
        // SelfAttention matches probe<->payload by content in ONE hop; an
        // LSTM must carry set-membership through its hidden state.
        Random gen = new Random(42L);
        int n = 6400;
        float[] X = new float[n * T * V];
        int[] Y = new int[n];
        int[][] seqs = new int[n][T];
        for (int i = 0; i < n; i++) {
            boolean present = gen.nextBoolean();
            int probe = gen.nextInt(V);
            seqs[i][0] = probe;
            Integer[] pay = distinctWithout(gen, T - 1, probe);
            seqs[i][0] = probe;
            for (int t = 1; t < T; t++) seqs[i][t] = pay[t - 1];
            if (present) seqs[i][1 + gen.nextInt(T - 1)] = probe;   // plant it
            Y[i] = present ? 1 : 0;
            for (int t = 0; t < T; t++)
                X[i * T * V + t * V + seqs[i][t]] = 1f;
        }

        int[][] parts = Datasets.trainTestSplit(n, 0.2, 7L);
        float[] trainX = Datasets.gatherRows(X, parts[0], T * V);
        float[] testX = Datasets.gatherRows(X, parts[1], T * V);
        int[] trainY = Datasets.gatherValues(Y, parts[0]);
        int[] testY = Datasets.gatherValues(Y, parts[1]);
        long presentN = java.util.Arrays.stream(testY).filter(y -> y == 1).count();
        System.out.printf("%d retrieval samples (%d%% probe-present); split %d/%d%n",
                n, Math.round(100.0 * presentN / testY.length),
                parts[0].length, parts[1].length);

        double accAttention, accLstm;

        // ---------- 2. Model A: two attention blocks ------------------------
        System.out.println("\n=== A: SelfAttention x2 (mini-transformer) ===");
        Optimizer optA = Optimizer.adam(0.02f);
        optA.setWeightDecay(1e-4f);
        try (Sequential net = new Sequential(optA, 42L,
                new Input3D(T, V),
                new SelfAttention(),
                new SelfAttention(),
                new Flatten(),
                new Dropout(0.2f),
                new Dense(24), new ReLU(),
                new Dense(2))) {
            net.setShuffleSeed(42L);
            EarlyStopping es = EarlyStopping.valLoss(30, true);
            AbstractClassifier.History h =
                    net.fit(trainX, trainY, epochs, 32, testX, testY, true, es);
            System.out.printf("ran %d/%d epochs - best val loss %.4f%n",
                    h.epochsRun(), epochs, es.bestScore);
            accAttention = report(net, testX, testY,
                    new String[]{"probe absent", "probe present"});
        }

        // ---------- 3. Model B: LSTM ----------------------------------------
        System.out.println("\n=== B: LSTM(32) ===");
        Optimizer optB = Optimizer.adam(0.02f);
        optB.setWeightDecay(1e-4f);
        try (Sequential net = new Sequential(optB, 43L,
                new Input3D(T, V),
                new LSTM(48),
                new Flatten(),
                new Dropout(0.2f),
                new Dense(24), new ReLU(),
                new Dense(2))) {
            net.setShuffleSeed(43L);
            EarlyStopping es = EarlyStopping.valLoss(30, true);
            AbstractClassifier.History h =
                    net.fit(trainX, trainY, epochs, 32, testX, testY, true, es);
            System.out.printf("ran %d/%d epochs - best val loss %.4f%n",
                    h.epochsRun(), epochs, es.bestScore);
            accLstm = report(net, testX, testY,
                    new String[]{"probe absent", "probe present"});
        }

        System.out.printf("%nTest accuracy: attention %.4f | lstm %.4f%n",
                accAttention, accLstm);
    }

    /* ------------------------- helper methods --------------------------- */


    static double report(Sequential net, float[] x, int[] y, String[] names) {
        float[] probs = net.predict(x);
        int correct = 0;
        double loss = 0;
        int[][] conf = new int[2][2];
        for (int i = 0; i < y.length; i++) {
            int pred = Metrics.argmax(probs, i * 2, 2);
            conf[y[i]][pred]++;
            if (pred == y[i]) correct++;
            loss += -Math.log(Math.max(probs[i * 2 + y[i]], 1e-12));
        }
        Metrics.printReport(loss / y.length, (double) correct / y.length,
                conf, names);
        return (double) correct / y.length;
    }

    /** count distinct digit ids, excluding the probe symbol. */
    static Integer[] distinctWithout(Random gen, int count, int probe) {
        java.util.List<Integer> cand = new java.util.ArrayList<>();
        for (int i = 0; i < V; i++) if (i != probe) cand.add(i);
        java.util.Collections.shuffle(cand, gen);
        return cand.subList(0, count).toArray(new Integer[0]);
    }
    private AttentionVsLstm() {}
}
