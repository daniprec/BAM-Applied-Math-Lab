function Header(el)
  if el.level <= 2 then
    return nil
  end

  return pandoc.Div(
    {
      pandoc.Plain({
        pandoc.Span(el.content, pandoc.Attr("", { "panel-title" }))
      })
    },
    pandoc.Attr("", { "panel-title-block" })
  )
end
