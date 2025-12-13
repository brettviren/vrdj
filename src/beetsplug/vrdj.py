'''
The vrdj plugin for beets
'''
from pathlib import Path
from beets.plugins import BeetsPlugin
from beets.ui import Subcommand, decargs, print_
from beets import config as main_config

class VrdjPlugin(BeetsPlugin):

    def __init__(self, name='vrdj'):
        super().__init__(name)
        self.config.add({
            'directory':'',
            'force':False,
            'auto':False,
            'device':'cpu',
            'embedding':'vggish',
            'metric':'cosine'})
        if self.config['auto'].get(bool):
            self.register_listener('item_imported', self.vrdj_ingest_item)
            self.register_listener('album_imported', self.vrdj_ingest_album)

    @property
    def vrdj_store(self):
        if not hasattr(self, '_vrdj_store'):
            from vrdj import db
            directory = self.config['directory'].get()
            if not directory:
                library_file_path = main_config['library'].as_filename()
                directory = Path(library_file_path).parent / "vrdj"
            else:
                directory = Path(directory)
            embedding = self.config['embedding'].get()
            metric = self.config['metric'].get()
            device = self.config['device'].get()
            # print(f'vrdj: {embedding=} {metric=} {device=} {directory=}')
            self._vrdj_store = db.Store(directory, metric=metric,
                                        embedding=embedding, device=device)
        return self._vrdj_store
            

    def vrdj_ingest_album(self, lib, album):
        for item in album.items():
            self.vrdj_ingest_item(item)

    def vrdj_ingest_item(self, lib, item):
        store = self.vrdj_store
        item_path = item.path.decode()
        self._log.debug(f'ingesting {item.id} {item_path}')
        try:
            store.add_embedding(item.id, item_path)
        except Exception as err:
            self._log.error(f'failed with {err}')
            self._log.error(f'is the file valid? {item_path}')
            return
        return item

    def commands(self):
        vrdj_command = Subcommand(
            'vrdj',
            help='Virtual Radio DJ',
        )
        vrdj_command.parser.add_option(
            '-P', '--playlist', type=str, default=None,
            help='Output results as an m3u playlist file')
        vrdj_command.parser.add_option(
            '-n', '--number', default=10, type=int,
            help='Max number of similar items')
        vrdj_command.parser.add_all_common_options()
        vrdj_command.parser.usage += (
            "\nSimilarity indexing and queries"
        )
        vrdj_command.func = self._vrdj_command_func
        return [vrdj_command]

    def _vrdj_command_func(self, lib, opts, args):

        from vrdj.op import similar_average_many

        query = decargs(args)
        items = lib.items(query)

        # ingest no matter what.  This is idempotent but we'll see if it is fast
        # enough to keep
        item_ids = list()
        for item in items:
            item = self.vrdj_ingest_item(lib, item)
            if item is None:
                continue
            item_ids.append(item.id)
        if not item_ids:
            self._log.error("no seed items")
            return

        new_ids = similar_average_many(self.vrdj_store, item_ids, opts.number)
        if not new_ids:
            self._log.error("no similar songs")

        if opts.playlist:
            out = open(opts.playlist, "w")
            out.write("#EXTM3U\n")
            for item_id in new_ids:
                item = lib.get_item(item_id)
                if item is None:
                    self._log.error(f'no item for {item_id=}')
                    continue
                out.write(f"#EXTINF:{int(item.length)},{item.artist} - {item.title}\n")
                out.write(item.path.decode() + "\n")

        for item_id in new_ids:
            #print(f'{item_id=} {type(item_id)}')
            item = lib.get_item(item_id)
            if item is None:
                self._log.error(f'no item for {item_id=}')
                continue
            print_(format(item))
            #print_(format(item, fmt))
            # print(f'{item=}')

        # if --ingest
        # if --playlist
        # else print

            
