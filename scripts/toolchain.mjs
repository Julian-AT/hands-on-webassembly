import {createHash} from 'node:crypto';
import {readFile, writeFile, mkdir, access, rm} from 'node:fs/promises';
import {spawnSync} from 'node:child_process';
import {resolve, dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const pins = JSON.parse(await readFile(resolve(root,'manifests/toolchain.json'),'utf8'));
const key = `${process.platform}-${process.arch}`;
const checksum = pins.node.archives[key];
if (!checksum) throw new Error(`Unsupported toolchain platform ${key}`);
const cache = resolve(root,'.tools'); await mkdir(cache,{recursive:true});
async function download(url, path, algorithm, digest, encoding='hex') {
  let data;
  try { data=await readFile(path); } catch { const response=await fetch(url); if(!response.ok) throw new Error(`Toolchain HTTP ${response.status}`); data=Buffer.from(await response.arrayBuffer()); }
  if(createHash(algorithm).update(data).digest(encoding)!==digest) throw new Error(`Toolchain checksum rejected: ${url}`);
  await writeFile(path,data);
}
function run(command,args,env={}) {
  const result=spawnSync(command,args,{cwd:root,stdio:'inherit',env:{...process.env,...env}});
  if(result.error) throw result.error;
  if(result.status!==0) process.exit(result.status||1);
}
const nodeName=`node-v${pins.node.version}-${key}`;
const nodeArchive=resolve(cache,`${nodeName}.tar.gz`), nodeDir=resolve(cache,nodeName);
await download(`https://nodejs.org/dist/v${pins.node.version}/${nodeName}.tar.gz`,nodeArchive,'sha256',checksum);
try { await access(resolve(nodeDir,'bin/node')); } catch { run('tar',['-xzf',nodeArchive,'-C',cache]); }
const node=resolve(nodeDir,'bin/node');
if(spawnSync(node,['--version'],{encoding:'utf8'}).stdout.trim()!==`v${pins.node.version}`) throw new Error('Wrong Node version');
const npmArchive=resolve(cache,`npm-${pins.npm.version}.tgz`), npmDir=resolve(cache,`npm-${pins.npm.version}`);
await download(`https://registry.npmjs.org/npm/-/npm-${pins.npm.version}.tgz`,npmArchive,'sha512',pins.npm.integrity.slice(7),'base64');
try { await access(resolve(npmDir,'package/bin/npm-cli.js')); } catch { await mkdir(npmDir,{recursive:true}); run('tar',['-xzf',npmArchive,'-C',npmDir]); }
const npm=resolve(npmDir,'package/bin/npm-cli.js');
const env={PATH:`${resolve(nodeDir,'bin')}:${process.env.PATH}`,npm_config_cache:resolve(cache,'npm-cache'),NEXT_TELEMETRY_DISABLED:'1'};
if(spawnSync(node,[npm,'--version'],{encoding:'utf8',env:{...process.env,...env}}).stdout.trim()!==pins.npm.version) throw new Error('Wrong npm version');
const mode=process.argv[2];
if(mode==='install') run(node,[npm,'ci','--no-audit','--no-fund'],env);
else if(mode==='lock') run(node,[npm,'install','--package-lock-only','--ignore-scripts','--no-audit','--no-fund'],env);
else if(mode==='build') run(node,['scripts/build.mjs'],env);
else if(mode==='test') run(node,['--test','tests/navigation.test.mjs','tests/archive.test.mjs','tests/launcher.test.mjs'],env);
else throw new Error('Use install, lock, build or test');
