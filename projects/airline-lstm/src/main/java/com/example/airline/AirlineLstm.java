package com.example.airline;

import com.example.tfcapi.*;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Project 8 - Airline passengers time series with an LSTM.
 *
 * Classic 1949-1960 monthly airline passenger counts (144 points). We turn the
 * series into a supervised problem: for every month t the label is the
 * direction of the next month-over-month change,
 *
 *     up   if p[t+1] > p[t]      down otherwise
 *
 * and the features are the last six deltas delta[t-5..t] (each
 * delta[i] = p[i+1]-p[i], early windows repeat their oldest available delta),
 * standardized with TRAIN-period statistics. That yields 143 examples which
 * are split CHRONOLOGICALLY - first 100 train, last 43 test - never shuffled,
 * because in time series the future must stay out of training.
 *
 * Model: Input3D(6,1) -> LSTM(8) -> Dense(16) -> ReLU -> Dense(2)
 * The LSTM emits the flattened [T*hidden] = [6*8] = 48 outputs per example.
 * A majority-class baseline is reported so the network's accuracy has context.
 */
public final class AirlineLstm {

    private static final int WINDOW = 6;
    private static final String[] NAMES = {"down", "up"};

    public static void main(String[] args) throws Exception {
        // ---------- 1. Load the CSV from resources ------------------------
        List<String> lines = readLines("/data/airline-passengers.csv");
        int nMonths = lines.size() - 1; // skip header "Month","Passengers"
        float[] p = new float[nMonths];
        String[] months = new String[nMonths];
        for (int i = 0; i < nMonths; i++) {
            // rows look like: "1949-01",112
            String[] parts = lines.get(i + 1).replace("\"", "").split(",");
            months[i] = parts[0];
            p[i] = Float.parseFloat(parts[1].trim());
        }
        System.out.println("Loaded " + nMonths + " monthly values (" + months[0]
                + " .. " + months[nMonths - 1] + ")");

        // ---------- 2. Month-over-month deltas ----------------------------
        int nDelta = nMonths - 1; // 143 transitions
        float[] delta = new float[nDelta];
        for (int i = 0; i < nDelta; i++) delta[i] = p[i + 1] - p[i];

        // ---------- 3. Chronological split BEFORE standardizing -----------
        // First 100 examples train, last 43 test. Statistics come from the
        // train period only, exactly as in the tabular projects.
        int nTrain = 100;
        int nTest = nDelta - nTrain; // 43

        // Train-period mean/std of deltas (used by both splits).
        double sum = 0;
        for (int i = 0; i < nTrain; i++) sum += delta[i];
        float mean = (float) (sum / nTrain);
        double sq = 0;
        for (int i = 0; i < nTrain; i++) { float d = delta[i] - mean; sq += d * d; }
        float std = (float) Math.sqrt(sq / Math.max(nTrain - 1, 1));
        System.out.printf("train delta mean=%.3f std=%.3f%n", mean, std);

        float[] z = new float[nDelta];
        for (int i = 0; i < nDelta; i++) z[i] = (delta[i] - mean) / (std + 1e-8f);

        // ---------- 4. Sliding windows -> [examples x 6] sequences ---------
        float[] trainX = new float[nTrain * WINDOW];
        int[] trainY = new int[nTrain];
        float[] testX = new float[nTest * WINDOW];
        int[] testY = new int[nTest];
        for (int t = 0; t < nDelta; t++) {
            // window holds z[t-5..t]; before index 5 we clamp to the start
            for (int k = 0; k < WINDOW; k++) {
                int idx = Math.max(t - (WINDOW - 1) + k, 0);
                if (t < nTrain) trainX[t * WINDOW + k] = z[idx];
                else testX[(t - nTrain) * WINDOW + k] = z[idx];
            }
            int label = delta[t] > 0 ? 1 : 0; // up if p[t+1] > p[t]
            if (t < nTrain) trainY[t] = label;
            else testY[t - nTrain] = label;
        }

        // ---------- 5. Majority-class baseline (sanity check) -------------
        int trainUps = 0;
        for (int y : trainY) trainUps += y;
        int majority = trainUps >= nTrain - trainUps ? 1 : 0;
        int baseCorrect = 0;
        for (int y : testY) baseCorrect += (y == majority ? 1 : 0);
        System.out.printf("Majority baseline (train class=%s): test accuracy %.3f (%d/%d)%n",
                NAMES[majority], (double) baseCorrect / nTest, baseCorrect, nTest);

        // ---------- 6. Build and train the LSTM ---------------------------
        try (Sequential net = new Sequential(Optimizer.adam(0.01f),
                new Input3D(WINDOW, 1),
                new LSTM(8),                 // output flattens to [batch, 6*8=48]
                new Dense(16), new ReLU(),
                new Dense(2))) {

            net.fit(trainX, trainY, 200, 16, testX, testY, true);

            // ---------- 7. Evaluate ---------------------------------------
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
            Metrics.printReport(loss / nTest, (double) correct / nTest, conf, NAMES);
            System.out.printf("Model vs baseline: %.3f vs %.3f%n",
                    (double) correct / nTest, (double) baseCorrect / nTest);

            net.saveTo(Path.of("target", "airline-lstm-model.bin"));
            System.out.println("Model saved to target/airline-lstm-model.bin");
        }
    }

    /* ------------------------- helper methods --------------------------- */

    static List<String> readLines(String resource) throws Exception {
        try (InputStream in = AirlineLstm.class.getResourceAsStream(resource)) {
            if (in == null) throw new IllegalStateException("resource not found: " + resource);
            List<String> out = new ArrayList<>();
            BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) out.add(line);
            }
            return out;
        }
    }

    private AirlineLstm() {}
}
