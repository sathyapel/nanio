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
  'packages/ihr',
  'packages/examples'
];

const newVersion = '0.0.1-alpha.2';

for (const pkg of packages) {
  const pjsonPath = path.join(pkg, 'package.json');
  const content = JSON.parse(fs.readFileSync(pjsonPath, 'utf8'));

  content.version = newVersion;

  if (content.dependencies) {
    for (const dep of Object.keys(content.dependencies)) {
      if (dep.startsWith('@nanio/')) {
        content.dependencies[dep] = `^${newVersion}`;
      }
    }
  }

  fs.writeFileSync(pjsonPath, JSON.stringify(content, null, 2) + '\n', 'utf8');
  console.log(`Updated ${pjsonPath} to version ${newVersion}`);
}
