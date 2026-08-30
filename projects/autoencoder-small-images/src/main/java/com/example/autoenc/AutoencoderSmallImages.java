package com.example.autoenc;

import com.example.tfcapi.*;

import java.nio.file.Path;
import java.util.Random;

/**
 * Project 12 - Autoencoder for small synthetic images (denoising variant).
 *
 * An autoencoder squeezes an image through a low-dimensional latent code and
 * reconstructs it; training simply regresses outputs against the INPUT images
 * themselves, which is why a Regressor (float targets, MSE loss) is the right
 * model - no class labels anywhere. Demonstrates:
 *
 * 1. generating a seeded dataset of 12x12 grayscale pattern images,
 * 2. Datasets.trainTestSplit / gatherRows for reproducible splits,
 * 3. an hourglass MLP (144 -> 72 -> 24 -> 72 -> 144) with BatchNorm1D,
 * 4. DENOISING: train on noise-corrupted inputs against clean targets,
 * 5. EarlyStopping.valLoss(15, true) + cosine LR schedule,
 * 6. RMSE/PSNR evaluation and ASCII previews of original vs reconstruction.
 */
public final class AutoencoderSmallImages {

    private static final int S = 12;                   // image is S x S
    private static final int PIX = S * S;              // 144 inputs/targets
    private static final int N = 1280;                 // 70% train split =
                                                       // 896 = 28 x batch 32
    private static final String[] NAMES = {"cross", "box", "disc"};

    public static void main(String[] args) throws Exception {
        int epochs = args.length > 0 ? Integer.parseInt(args[0]) : 150;

        // ---------- 1. Generate the synthetic pattern images --------------
        Random gen = new Random(42L);
        float[] cleanX = new float[N * PIX];
        int[] type = new int[N];
        for (int i = 0; i < N; i++) {
            type[i] = gen.nextInt(NAMES.length);
            int dy = gen.nextInt(3) - 1;               // jitter: shift +-1 px
            int dx = gen.nextInt(3) - 1;
            draw(type[i], dy, dx, cleanX, i * PIX);
            for (int p = 0; p < PIX; p++) {            // sensor noise
                float v = cleanX[i * PIX + p] + (float) gen.nextGaussian() * 0.05f;
                cleanX[i * PIX + p] = Math.max(0f, Math.min(1f, v));
            }
        }

        // ---------- 2. Seeded split via Datasets --------------------------
        int[][] parts = Datasets.trainTestSplit(N, 0.3, 7L);
        int[] trIdx = parts[0], teIdx = parts[1];
        float[] trainClean = Datasets.gatherRows(cleanX, trIdx, PIX);
        float[] testClean = Datasets.gatherRows(cleanX, teIdx, PIX);
        int[] testType = new int[teIdx.length];
        for (int i = 0; i < teIdx.length; i++) testType[i] = type[teIdx[i]];
        System.out.printf("Generated %d %dx%d images (%s); split %d/%d%n",
                N, S, S, String.join("/", NAMES), trIdx.length, teIdx.length);

        // ---------- 3. Corrupt the INPUTS only (targets stay clean) -------
        float[] trainNoisy = addNoise(trainClean, 0.25f, 123L);
        float[] testNoisy = addNoise(testClean, 0.25f, 124L);

        // ---------- 4. Hourglass autoencoder trained as regression --------
        // Reconstruction favors unshrunk weights, so no weight decay here -
        // just gradient clipping as a safety net. BatchNorm sits between the
        // Dense and its ReLU (the canonical position).
        Optimizer opt = Optimizer.adam(0.01f);
        opt.setGradClipNorm(5f);

        try (Regressor net = new Regressor(opt, 42L,
                new Input(PIX),
                new Dense(64), new BatchNorm1D(), new ReLU(),
                new Dense(24), new ReLU(),                 // latent code
                new Dense(64), new BatchNorm1D(), new ReLU(),
                new Dense(144))) {                         // linear output

            net.setShuffleSeed(42L);
            net.setLrSchedule(LrSchedule.cosine(epochs, 0.05f));

            // Monitor val loss on NOISY inputs vs CLEAN targets, so early
            // stopping tracks exactly the skill we care about (denoising).
            EarlyStopping es = EarlyStopping.valLoss(15, true);
            Regressor.RegressionHistory h =
                    net.fit(trainNoisy, trainClean, epochs, 32,
                            testNoisy, testClean, true, es);
            System.out.printf("Ran %d/%d epochs - best val MSE %.6f at epoch %d%n",
                    h.epochsRun(), epochs, es.bestScore, es.bestEpoch + 1);

            net.setTraining(false);                        // BN running stats

            // ---------- 5. Evaluate both skills ---------------------------
            double mseNoisy = evalMse(net, testNoisy, testClean);
            double psnrNoisy = 10 * Math.log10(1.0 / mseNoisy);
            System.out.printf("Denoising  : RMSE %.4f | PSNR %.2f dB%n",
                    Math.sqrt(mseNoisy), psnrNoisy);

            float[] rec = net.predict(testClean);          // plain reconstruction
            double mseRec = 0;
            for (int i = 0; i < rec.length; i++) {
                double d = rec[i] - testClean[i];
                mseRec += d * d;
            }
            mseRec /= rec.length;
            System.out.printf("Reconstruct: RMSE %.4f | PSNR %.2f dB%n%n",
                    Math.sqrt(mseRec), 10 * Math.log10(1.0 / mseRec));

            // ---------- 6. ASCII previews ----------------------------------
            for (int k = 0; k < 3 && k < teIdx.length; k++) {
                System.out.println("test image " + (k + 1) + " (" +
                        NAMES[testType[k]] + ")   original | denoised");
                ascii(testClean, k, rec, k);
            }

            net.saveTo(Path.of("target", "autoencoder.bin"));
            System.out.println("Saved checkpoint target/autoencoder.bin");
        }
    }

    /* ------------------------- helper methods --------------------------- */

    /** Draws one of three patterns into img at [off], shifted by (dy,dx). */
    static void draw(int t, int dy, int dx, float[] img, int off) {
        for (int r = 0; r < S; r++) {
            for (int c = 0; c < S; c++) {
                float v = 0.08f;                           // dim background
                int rr = r - dy, cc = c - dx;              // unshifted coords
                switch (t) {
                    case 0 -> {                            // diagonal cross
                        if (rr >= 1 && rr <= 10 &&
                            (rr == cc || rr + cc == 11)) v = 0.95f;
                    }
                    case 1 -> {                            // rectangle outline
                        if ((rr == 2 || rr == 9) && cc >= 2 && cc <= 9)
                            v = 0.9f;
                        if ((cc == 2 || cc == 9) && rr >= 2 && rr <= 9)
                            v = 0.9f;
                    }
                    default -> {                           // filled disc
                        if (dist2(rr, cc, 5.5, 5.5) <= 3.2 * 3.2) v = 0.85f;
                    }
                }
                img[off + r * S + c] = v;
            }
        }
    }

    static double dist2(double r, double c, double cr, double cc) {
        return (r - cr) * (r - cr) + (c - cc) * (c - cc);
    }

    /** Returns a copy of data plus Gaussian noise, clamped to [0,1]. */
    static float[] addNoise(float[] data, float sigma, long seed) {
        Random r = new Random(seed);
        float[] out = data.clone();
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.max(0f, Math.min(1f, out[i]
                    + (float) r.nextGaussian() * sigma));
        }
        return out;
    }

    static double evalMse(Regressor net, float[] x, float[] y) {
        float[] p = net.predict(x);
        double sum = 0;
        for (int i = 0; i < p.length; i++) {
            double d = p[i] - y[i];
            sum += d * d;
        }
        return sum / p.length;
    }

    /** Prints two 12x12 images side by side as ASCII art. */
    private static void ascii(float[] a, int ia, float[] b, int ib) {
        String ramp = " .:-=+*#@";
        for (int r = 0; r < S; r++) {
            StringBuilder line = new StringBuilder();
            for (int c = 0; c < S; c++) {
                line.append(ramp.charAt(clampIdx(a[(ia * PIX) + r * S + c])));
            }
            line.append("  |  ");
            for (int c = 0; c < S; c++) {
                line.append(ramp.charAt(clampIdx(b[(ib * PIX) + r * S + c])));
            }
            System.out.println(line);
        }
        System.out.println();
    }

    /** Maps any value (reconstructions can leave [0,1]) to a ramp index. */
    private static int clampIdx(float v) {
        return Math.max(0, Math.min(8, (int) (v * 9f)));   // ramp has 9 chars
    }

    private AutoencoderSmallImages() {}
}
