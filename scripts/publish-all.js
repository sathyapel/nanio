import { execSync } from 'child_process';
import readline from 'readline';

const packages = [
  'packages/tools',
  'packages/embeddings',
  'packages/core',
  'packages/observability',
  'packages/registry',
  'packages/vectorstore',
  'packages/providers',
  'packages/ihr'
];

function promptOtp() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  return new Promise((resolve) => {
    rl.question('Enter NPM 2FA One-Time Password (OTP): ', (answer) => {
      rl.close();
      resolve(answer.trim());
    });
  });
}

async function main() {
  const otp = await promptOtp();
  const otpFlag = otp ? ` --otp=${otp}` : '';

  for (const pkg of packages) {
    console.log(`\nPublishing ${pkg}...`);
    try {
      execSync(`npm publish --access public --tag alpha${otpFlag}`, {
        cwd: pkg,
        stdio: 'inherit'
      });
      console.log(`Successfully published ${pkg}`);
    } catch (error) {
      console.error(`Failed to publish ${pkg}:`, error.message);
      process.exit(1);
    }
  }
  console.log('\nAll packages published successfully!');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
