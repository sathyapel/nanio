import { execSync } from 'child_process';
import readline from 'readline';
import fs from 'fs';
import path from 'path';

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

const allPackageDirs = [
  ...packages,
  'packages/examples'
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

function bumpVersions(newVersion) {
  console.log(`\nBumping package versions to: ${newVersion}`);
  
  // 1. Update root package.json version
  const rootPkgPath = './package.json';
  if (fs.existsSync(rootPkgPath)) {
    const rootPkg = JSON.parse(fs.readFileSync(rootPkgPath, 'utf8'));
    rootPkg.version = newVersion;
    fs.writeFileSync(rootPkgPath, JSON.stringify(rootPkg, null, 2) + '\n', 'utf8');
    console.log(`Updated root package.json version`);
  }

  // 2. Update each package package.json
  for (const dir of allPackageDirs) {
    const pkgPath = path.join(dir, 'package.json');
    if (!fs.existsSync(pkgPath)) continue;
    
    const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
    pkg.version = newVersion;

    // Update package description to match new version suffix (v0.0.1-alpha.X)
    if (pkg.description) {
      pkg.description = pkg.description.replace(/\(v\d+\.\d+\.\d+-alpha\.\d+\)/, `(v${newVersion})`);
    }
    
    // Update internal dependencies starting with @nanio/
    if (pkg.dependencies) {
      for (const dep of Object.keys(pkg.dependencies)) {
        if (dep.startsWith('@nanio/')) {
          pkg.dependencies[dep] = `^${newVersion}`;
        }
      }
    }
    if (pkg.devDependencies) {
      for (const dep of Object.keys(pkg.devDependencies)) {
        if (dep.startsWith('@nanio/')) {
          pkg.devDependencies[dep] = `^${newVersion}`;
        }
      }
    }
    if (pkg.peerDependencies) {
      for (const dep of Object.keys(pkg.peerDependencies)) {
        if (dep.startsWith('@nanio/')) {
          pkg.peerDependencies[dep] = `^${newVersion}`;
        }
      }
    }
    
    fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + '\n', 'utf8');
    console.log(`Updated ${pkgPath}`);
  }

  // 3. Re-install to update package-lock.json and rebuild
  console.log('\nRunning npm install to update package lock...');
  execSync('npm install', { stdio: 'inherit' });
  
  console.log('\nBuilding packages...');
  execSync('npm run build', { stdio: 'inherit' });
}

async function main() {
  const newVersion = process.argv[2];
  if (newVersion) {
    bumpVersions(newVersion);
  } else {
    console.log('No version specified. Skipping version bump. Pass a version like "node scripts/publish-all.js 0.0.1-alpha.2" to bump versions before publishing.');
  }

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
