import click
import os
from pathlib import Path
import logging
import numpy as np
import random
from typing import List
import yaml

# Set up simple logging for the CLI
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
_log = logging.getLogger(__name__)

class Main:
    def __init__(self, directory, scheme, device):
        self._directory = directory
        self._scheme = scheme
        self._device = device

    @property
    def store(self):
        if not hasattr(self, '_store'):
            from vrdj import db
            # lazy load to lower latency for things not needing the store
            if not self._directory:
                from vrdj import beetface
                self._directory = (beetface.dbpath().parent / "vrdj").absolute()
            self._store = db.Store(self._directory,
                                   scheme=self._scheme,
                                   device=self._device)
        return self._store

@click.group()
@click.option('-d', '--directory',
              default=None,
              help='Path to the vrdj store directory.',
              type=click.Path(exists=True, file_okay=False, dir_okay=True,
                              resolve_path=True, writable=True, path_type=Path))
@click.option('-s', '--scheme', default='vggish-mean-cosine',
              help='Indexing scheme (e.g., vggish-mean-cosine).')
@click.option('--device', default='cpu',
              help='Device for torch',
              type=click.Choice(["cpu","cuda"])) # fixme: add more
@click.pass_context
def cli(ctx, directory, scheme, device):
    """
    Virtual Radio DJ (VRDJ) CLI for indexing and searching audio similarity 
    based on VGGish embeddings and Faiss.
    """
    ctx.obj = Main(directory, scheme, device)


@cli.command('beets')
@click.pass_context
def beets(ctx):
    from vrdj import beetface
    print(ctx.obj.store.dirpath)
    print(f'{beetface.dbpath()=}')
    print(f'{beetface.music_directory()=}')

@cli.command('ingest')
@click.argument('filepaths', nargs=-1, type=click.Path(exists=True))
@click.pass_context
def cmd_ingest(ctx, filepaths):
    '''
    Ingest files into the index.
    '''
    from vrdj import op, beetface
    lib = beetface.library()
    for item_path in filepaths:
        print(f'ingesting {item_path}')
        item = beetface.item_at_path(lib, item_path)
        if not item:
            print(f'failed to get {item_path}')
            continue
        item_path = item.path.decode()
        print(f'ingesting {item.id} {item_path}')
        op.ingest(ctx.obj.store, item_path, item.id)

def main():
    cli(obj={})
    
